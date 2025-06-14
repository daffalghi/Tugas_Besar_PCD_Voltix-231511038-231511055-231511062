# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from pathlib import Path
import settings
import io
from PIL import Image
import base64
import time
import json
import asyncio

# Impor RKNN API
from rknn.api import RKNN # Penting: Pastikan rknn-toolkit-lite terinstal di Orange Pi

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Variabel global untuk instance RKNN
rknn_instance = None
RKNN_MODEL_FILE = Path(settings.DETECTION_MODEL)

# Event untuk memberi sinyal stop ke webcam stream
webcam_stop_event = asyncio.Event()

# --- Fungsi Pemuatan dan Pelepasan Model RKNN ---
@app.on_event("startup")
async def startup_event():
    global rknn_instance
    print(f"--> Memuat model RKNN: {RKNN_MODEL_FILE}")
    
    rknn_instance = RKNN(verbose=True) # Set verbose=False untuk output lebih ringkas
    
    # Memuat model RKNN dari file
    ret = rknn_instance.load_rknn(str(RKNN_MODEL_FILE))
    if ret != 0:
        print(f"Gagal memuat model RKNN! Kode return: {ret}")
        # Jika model gagal dimuat, aplikasi mungkin tidak bisa berfungsi.
        # Lempar Exception agar FastAPI tidak jalan jika model esensial.
        raise Exception(f"Gagal memuat model RKNN dari {RKNN_MODEL_FILE}. Pastikan model valid dan RKNN Toolkit Lite terinstal dengan benar.")
    print("Selesai memuat model.")

    # Inisialisasi runtime environment
    print("--> Menginisialisasi runtime RKNN...")
    # async_mode=False lebih sederhana untuk memulai. async_mode=True jika butuh inferensi non-blocking yang kompleks.
    ret = rknn_instance.init_runtime(async_mode=False) 
    if ret != 0:
        print(f"Gagal inisialisasi runtime RKNN! Kode return: {ret}")
        raise Exception("Gagal menginisialisasi runtime RKNN. Pastikan perangkat keras dan driver siap.")
    print("Runtime RKNN siap.")

@app.on_event("shutdown")
async def shutdown_event():
    global rknn_instance
    if rknn_instance:
        print("--> Melepaskan instance RKNN...")
        rknn_instance.release()
        print("Instance RKNN dilepaskan.")

# --- Fungsi Preprocessing dan Postprocessing untuk RKNN ---

def preprocess_image_for_rknn(image_np):
    """
    Melakukan preprocessing gambar untuk input model RKNN.
    Gambar harus dalam format BGR dari OpenCV.
    """
    # Mengubah BGR ke RGB (model YOLOv8 umumnya dilatih dengan RGB)
    img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    # Resize ke ukuran input model yang diharapkan (misal: 640x640)
    img_resized = cv2.resize(img_rgb, 
                             (settings.MODEL_INPUT_WIDTH, settings.MODEL_INPUT_HEIGHT), 
                             interpolation=cv2.INTER_LINEAR)
    
    # Menambahkan dimensi batch (1, H, W, C). RKNN Toolkit biasanya mengharapkan HWC.
    input_data = np.expand_dims(img_resized, axis=0).astype(np.float32)
    
    # Normalisasi: RKNN Toolkit biasanya menangani ini berdasarkan mean_values dan std_values
    # yang dikonfigurasi saat konversi. Jika tidak, Anda mungkin perlu melakukan normalisasi
    # secara eksplisit di sini (misalnya: input_data /= 255.0).
    return input_data

def postprocess_yolov8_rknn_output(rknn_outputs, original_img_shape):
    """
    Melakukan post-processing pada output mentah dari model RKNN YOLOv8.
    Ini termasuk decoding bounding box, scores, dan Non-Maximum Suppression (NMS).
    """
    # RKNN output adalah list of numpy arrays.
    # Untuk YOLOv8, output deteksi biasanya tensor tunggal.
    # Format ini sangat tergantung pada bagaimana model Anda diekspor/dikonversi.
    # Seringkali, outputnya adalah (1, num_det, 5 + num_classes) atau (1, 5 + num_classes, num_det).
    # Anda mungkin perlu menyesuaikan berdasarkan `rknn_outputs[0].shape` saat debugging.
    
    # Asumsi umum untuk YOLOv8 ONNX/RKNN output: (1, num_values, num_detections)
    # where num_values = 4 (bbox) + 1 (objectness) + num_classes
    predictions = rknn_outputs[0] # Ambil tensor output pertama
    
    # Transpose jika perlu untuk mendapatkan (num_detections, 5 + num_classes)
    if predictions.shape[1] == (5 + len(settings.ALL_CLASSES)):
        # Jika bentuknya (1, 5+num_classes, num_detections)
        predictions = predictions[0].T # Transpose ke (num_detections, 5+num_classes)
    elif predictions.shape[0] == 1:
        # Jika bentuknya (1, num_detections, 5+num_classes)
        predictions = predictions[0] # Cukup ambil batch pertama
    else:
        print(f"Peringatan: Bentuk output RKNN tidak dikenal: {predictions.shape}")
        return [], [], [], original_img_shape # Kembalikan kosong jika bentuk tidak sesuai

    boxes = []
    confidences = []
    class_ids = []

    # Iterasi setiap deteksi
    for detection in predictions:
        # Asumsi: [center_x, center_y, width, height, objectness_score, class_score_1, class_score_2, ...]
        bbox_data = detection[:4]
        obj_conf = detection[4]
        class_scores = detection[5:]

        # Menggabungkan objectness confidence dengan class scores
        scores = obj_conf * class_scores

        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence >= settings.CONF_THRESHOLD:
            center_x, center_y, width, height = bbox_data

            # Konversi dari (center_x, center_y, width, height) ke (x1, y1, x2, y2)
            x1 = int(center_x - width / 2)
            y1 = int(center_y - height / 2)
            x2 = int(center_x + width / 2)
            y2 = int(center_y + height / 2)

            boxes.append([x1, y1, x2, y2])
            confidences.append(float(confidence))
            class_ids.append(class_id)
    
    # Aplikasikan Non-Maximum Suppression (NMS)
    # cv2.dnn.NMSBoxes membutuhkan format (x, y, w, h) untuk kotak.
    # Kita perlu mengubah format kotak untuk NMS
    nms_boxes = [[x, y, w - x, h - y] for x, y, w, h in boxes]
    
    indices = cv2.dnn.NMSBoxes(nms_boxes, confidences, settings.CONF_THRESHOLD, settings.NMS_IOU_THRESHOLD)
    
    final_boxes = []
    final_confidences = []
    final_class_ids = []

    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            final_boxes.append(boxes[i])
            final_confidences.append(confidences[i])
            final_class_ids.append(class_ids[i])

    # Sesuaikan ukuran bounding box ke dimensi gambar asli
    original_h, original_w = original_img_shape
    input_h, input_w = settings.MODEL_INPUT_HEIGHT, settings.MODEL_INPUT_WIDTH

    scale_x = original_w / input_w
    scale_y = original_h / input_h

    scaled_boxes = []
    for box in final_boxes:
        x1, y1, x2, y2 = box
        scaled_x1 = int(x1 * scale_x)
        scaled_y1 = int(y1 * scale_y)
        scaled_x2 = int(x2 * scale_x)
        scaled_y2 = int(y2 * scale_y)
        scaled_boxes.append([scaled_x1, scaled_y1, scaled_x2, scaled_y2])

    return scaled_boxes, final_confidences, final_class_ids, original_img_shape

def draw_boxes_on_image(image_np, boxes, confidences, class_ids):
    """
    Menggambar bounding box dan label pada gambar.
    image_np diharapkan dalam format BGR.
    """
    names = settings.ALL_CLASSES 

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        confidence = confidences[i]
        class_id = class_ids[i]
        class_name = names[class_id]

        color = (0, 255, 0) # Warna hijau untuk bounding box
        text = f"{remove_dash_from_class_name(class_name)}: {confidence:.2f}"

        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_np, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image_np

# --- Fungsi Klasifikasi Sampah ---
def classify_waste_type(detected_items):
    """Mengklasifikasikan item yang terdeteksi ke dalam kategori sampah."""
    # Anda perlu mendefinisikan kategori ini di settings.py juga atau di sini
    # Contoh kategori:
    RECYCLABLE = {'aluminium', 'cardboard', 'glass', 'paper', 'plastic'}
    NON_RECYCLABLE = {'non_recyclable_fabric', 'food_waste', 'other_waste'}
    HAZARDOUS = {'battery', 'hazardous_chemical_waste'} # Tambahkan kelas berbahaya Anda

    recyclable_items = set(detected_items) & RECYCLABLE
    non_recyclable_items = set(detected_items) & NON_RECYCLABLE
    hazardous_items = set(detected_items) & HAZARDOUS
    return recyclable_items, non_recyclable_items, hazardous_items

def remove_dash_from_class_name(class_name):
    """Membersihkan nama kelas untuk tampilan di frontend."""
    return class_name.replace("_", " ")

# --- Endpoint FastAPI ---

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        # Convert ke BGR untuk OpenCV, karena OpenCV memuat dalam BGR
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) 
        
        original_h, original_w, _ = image_np.shape # Simpan bentuk asli untuk post-processing

        # Preprocess gambar untuk input RKNN
        processed_image = preprocess_image_for_rknn(image_np.copy()) # Gunakan copy agar original tidak berubah

        # Jalankan inferensi RKNN
        if rknn_instance is None:
            raise HTTPException(status_code=500, detail="Model RKNN belum dimuat atau inisialisasi gagal.")
        rknn_outputs = rknn_instance.inference(inputs=[processed_image])

        # Post-process output RKNN untuk mendapatkan kotak, confidence, dan class IDs
        boxes, confidences, class_ids, _ = postprocess_yolov8_rknn_output(
            rknn_outputs, (original_h, original_w)
        )
        
        # Dapatkan nama kelas yang terdeteksi
        detected_items = set([settings.ALL_CLASSES[class_id] for class_id in class_ids])

        # Klasifikasikan jenis sampah
        recyclable_items, non_recyclable_items, hazardous_items = classify_waste_type(detected_items)
        
        result_dict = {
            "recyclable": [remove_dash_from_class_name(item) for item in recyclable_items],
            "non_recyclable": [remove_dash_from_class_name(item) for item in non_recyclable_items],
            "hazardous": [remove_dash_from_class_name(item) for item in hazardous_items]
        }

        # Gambar bounding box pada gambar asli
        res_plotted = draw_boxes_on_image(image_np.copy(), boxes, confidences, class_ids)
        _, buffer = cv2.imencode(".jpg", res_plotted)
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        return {"results": result_dict, "image": encoded_image}
    except Exception as e:
        print(f"Error selama deteksi gambar: {e}") # Debugging
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat mendeteksi gambar: {e}")

@app.get("/video_feed")
async def video_feed():
    global latest_webcam_results
    webcam_stop_event.clear() # Reset event ketika stream baru dimulai
    print("Backend: Aliran webcam dimulai, event stop direset.")

    async def generate():
        cap = cv2.VideoCapture(settings.WEBCAM_PATH)
        if not cap.isOpened():
            print("Backend: Error: Tidak dapat membuka webcam.")
            # Kirim frame kosong jika kamera tidak dapat dibuka
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes() + b'\r\n')
            return

        print("Backend: Webcam berhasil dibuka. Streaming frame...")
        try:
            while True:
                # Periksa apakah event stop telah diset
                if webcam_stop_event.is_set():
                    print("Backend: Event stop webcam terdeteksi. Keluar dari loop.")
                    break # Keluar dari loop

                success, frame = cap.read()
                if not success:
                    print("Backend: Error: Gagal membaca frame dari webcam. Keluar dari loop.")
                    break

                original_h, original_w, _ = frame.shape

                # Preprocess frame untuk RKNN
                processed_frame = preprocess_image_for_rknn(frame.copy())

                # Jalankan inferensi RKNN
                if rknn_instance is None:
                    print("Backend: Model RKNN tidak dimuat selama streaming video. Melewati inferensi.")
                    # Jika model tidak siap, kirim frame tanpa deteksi atau frame kosong
                    _, buffer = cv2.imencode(".jpg", frame) 
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    await asyncio.sleep(0.05)
                    continue # Lanjutkan ke frame berikutnya

                rknn_outputs = rknn_instance.inference(inputs=[processed_frame])

                # Post-process output RKNN
                boxes, confidences, class_ids, _ = postprocess_yolov8_rknn_output(
                    rknn_outputs, (original_h, original_w)
                )
                
                detected_items = set([settings.ALL_CLASSES[class_id] for class_id in class_ids])

                # Klasifikasikan jenis sampah
                recyclable_items, non_recyclable_items, hazardous_items = classify_waste_type(detected_items)
                
                # Perbarui variabel hasil global
                latest_webcam_results["recyclable"] = [remove_dash_from_class_name(item) for item in recyclable_items]
                latest_webcam_results["non_recyclable"] = [remove_dash_from_class_name(item) for item in non_recyclable_items]
                latest_webcam_results["hazardous"] = [remove_dash_from_class_name(item) for item in hazardous_items]

                # Gambar bounding box pada frame
                res_plotted = draw_boxes_on_image(frame.copy(), boxes, confidences, class_ids)
                _, buffer = cv2.imencode(".jpg", res_plotted)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                await asyncio.sleep(0.01) # Jeda kecil agar feed lebih halus
        finally:
            # Pastikan cap.release() terpanggil bahkan jika ada error
            if cap.isOpened():
                cap.release()
                print("Backend: Webcam dilepaskan.")
            else:
                print("Backend: Webcam tidak dibuka, tidak ada yang dilepaskan.")

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace;boundary=frame")

@app.post("/stop_webcam_backend")
async def stop_webcam_backend():
    webcam_stop_event.set() # Set event untuk memberi sinyal stop pada stream
    print("Backend: Menerima sinyal stop dari frontend. Event diset.")
    return JSONResponse(content={"message": "Sinyal stop webcam terkirim."})

@app.get("/webcam_classification")
async def get_webcam_classification():
    return JSONResponse(content=latest_webcam_results)