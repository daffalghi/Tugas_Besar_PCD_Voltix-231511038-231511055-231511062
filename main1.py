from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from rknn.api import RKNN
import cv2
import numpy as np
from pathlib import Path
import settings
import io
from PIL import Image
import base64
import asyncio
import time # Import the time module

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize RKNN model
model_path = Path(settings.DETECTION_MODEL)
try:
    rknn = RKNN(verbose=True)
    model_path_str = str(model_path)
    if rknn.load_rknn(model_path_str) != 0:
        raise Exception(f"Failed to load RKNN model from {model_path_str}")
    if rknn.init_runtime(target='rk3588') != 0:
        raise Exception("Failed to initialize RKNN runtime")
except Exception as ex:
    raise Exception(f"Unable to load RKNN model. Check the specified path: {model_path} - {ex}")

latest_webcam_results = {
    "recyclable": [],
    "non_recyclable": [],
    "hazardous": []
}

webcam_stop_event = asyncio.Event()

def classify_waste_type(detected_items):
    recyclable_items = set(detected_items) & set(settings.RECYCLABLE)
    non_recyclable_items = set(detected_items) & set(settings.NON_RECYCLABLE)
    hazardous_items = set(detected_items) & set(settings.HAZARDOUS)
    return recyclable_items, non_recyclable_items, hazardous_items

def remove_dash_from_class_name(class_name):
    return class_name.replace("_", " ")

def preprocess_image_for_rknn(image_np):
    """Preprocess image for RKNN model input using letterboxing."""
    img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    original_h, original_w = img_rgb.shape[:2]
    
    # Calculate scaling factors
    scale = min(settings.MODEL_INPUT_WIDTH / original_w, settings.MODEL_INPUT_HEIGHT / original_h)
    
    # New dimensions after scaling (maintaining aspect ratio)
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    
    # Resize image
    resized_img = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create a blank canvas for padding (usually grey 128)
    padded_img = np.full((settings.MODEL_INPUT_HEIGHT, settings.MODEL_INPUT_WIDTH, 3), 128, dtype=np.uint8) 
    
    # Calculate padding offsets
    x_offset = (settings.MODEL_INPUT_WIDTH - new_w) // 2
    y_offset = (settings.MODEL_INPUT_HEIGHT - new_h) // 2
    
    # Place the resized image onto the center of the padded canvas
    padded_img[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_img
    
    input_data = np.expand_dims(padded_img, axis=0).astype(np.float32)
    
    print(f"DEBUG PREPROCESS - Original: {original_w}x{original_h}, Scaled: {new_w}x{new_h}, Padded: {padded_img.shape}, x_offset: {x_offset}, y_offset: {y_offset}")
    return input_data

def decode_yolov8_boxes(boxes, anchors, stride, img_size):
    """Decode YOLOv8 anchor-based boxes to absolute coordinates."""
    boxes = boxes.copy()
    boxes[:, 0] = (boxes[:, 0] * stride)
    boxes[:, 1] = (boxes[:, 1] * stride)
    boxes[:, 2] = boxes[:, 2] * anchors[:, 0]
    boxes[:, 3] = boxes[:, 3] * anchors[:, 1]
    
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    x1 = np.clip(x1, 0, img_size[1])
    y1 = np.clip(y1, 0, img_size[0])
    x2 = np.clip(x2, 0, img_size[1])
    y2 = np.clip(y2, 0, img_size[0])
    
    return np.stack([x1, y1, x2, y2], axis=-1)

import numpy as np
import cv2
import settings 

def postprocess_yolov8_rknn_output(rknn_outputs, original_img_shape):
    print(f"DEBUG POSTPROCESS - RKNN outputs shapes: {[output.shape for output in rknn_outputs]}")

    predictions = rknn_outputs[0]
    predictions = predictions.transpose(0, 2, 1)
    predictions = predictions[0] 
    print(f"DEBUG POSTPROCESS - Predictions shape after transpose: {predictions.shape}")

    num_classes = len(settings.ALL_CLASSES)
    img_h, img_w = original_img_shape
    input_h, input_w = settings.MODEL_INPUT_HEIGHT, settings.MODEL_INPUT_WIDTH

    boxes_raw = predictions[:, :4] 
    scores = predictions[:, 4:] 

    max_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)

    mask = max_scores >= settings.CONF_THRESHOLD
    boxes = boxes_raw[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        print("No detections after confidence filtering.")
        return [], [], [], original_img_shape

    scale = min(input_w / img_w, input_h / img_h)

    unpadded_w_in_model_coords = int(img_w * scale)
    unpadded_h_in_model_coords = int(img_h * scale)

    x_offset_in_model_coords = (input_w - unpadded_w_in_model_coords) // 2
    y_offset_in_model_coords = (input_h - unpadded_h_in_model_coords) // 2

    final_boxes_on_original = []
    for box in boxes:
        cx_padded, cy_padded, w_padded, h_padded = box

        cx_unpadded = cx_padded - x_offset_in_model_coords
        cy_unpadded = cy_padded - y_offset_in_model_coords

        cx_original = cx_unpadded / scale
        cy_original = cy_unpadded / scale
        w_original = w_padded / scale
        h_original = h_padded / scale

        x1_original = cx_original - (w_original / 2)
        y1_original = cy_original - (h_original / 2)
        x2_original = cx_original + (w_original / 2)
        y2_original = cy_original + (h_original / 2)

        x1_original = np.clip(x1_original, 0, img_w)
        y1_original = np.clip(y1_original, 0, img_h)
        x2_original = np.clip(x2_original, 0, img_w)
        y2_original = np.clip(y2_original, 0, img_h)

        final_boxes_on_original.append([int(x1_original), int(y1_original), int(x2_original), int(y2_original)])

    nms_boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in final_boxes_on_original]

    indices = []
    if len(nms_boxes) > 0:
        indices = cv2.dnn.NMSBoxes(nms_boxes, max_scores.tolist(), settings.CONF_THRESHOLD, settings.NMS_IOU_THRESHOLD)

    final_boxes = []
    final_confidences = []
    final_class_ids = []

    if len(indices) > 0:
        indices = indices.flatten()
        final_boxes = [final_boxes_on_original[i] for i in indices]
        final_confidences = max_scores[indices]
        final_class_ids = class_ids[indices]

    detected_items = [settings.ALL_CLASSES[cls_id] for cls_id in final_class_ids if cls_id < num_classes]
    print(f"DEBUG POSTPROCESS - Detected items: {detected_items}, Final boxes count: {len(final_boxes)}")
    if final_boxes:
        print(f"DEBUG POSTPROCESS - Example box after scaling: {final_boxes[0]}")

    return final_boxes, final_confidences, final_class_ids, original_img_shape

def draw_boxes_on_image(image_np, boxes, confidences, class_ids, fps_text=""):
    """Draw detection boxes and FPS on image."""
    image_np = image_np.copy()
    names = settings.ALL_CLASSES
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        confidence = confidences[i]
        class_id = class_ids[i]
        if class_id < len(names):
            class_name = names[class_id]
            color = (0, 255, 0)
            text = f"{remove_dash_from_class_name(class_name)}: {confidence:.2f}"
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_np, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add FPS text
    cv2.putText(image_np, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return image_np

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        print("File diterima, ukuran:", len(contents))
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        print("Gambar berhasil dibaca, ukuran:", image.size)
        image_np = np.array(image)
        print("Array gambar:", image_np.shape)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        original_shape = image_np.shape[:2]
        print("Bentuk asli gambar:", original_shape)
        
        input_image = preprocess_image_for_rknn(image_np)
        print("Input preprocess selesai:", input_image.shape)
        outputs = rknn.inference(inputs=[input_image])
        print("Inferensi selesai, jumlah output:", len(outputs))
        boxes, confidences, class_ids, _ = postprocess_yolov8_rknn_output(outputs, original_shape)
        print("Hasil postprocessing:", len(boxes), "kotak terdeteksi")
        
        detected_items = [settings.ALL_CLASSES[cls_id] for cls_id in class_ids if cls_id < len(settings.ALL_CLASSES)]
        recyclable_items, non_recyclable_items, hazardous_items = classify_waste_type(detected_items)
        
        result_dict = {
            "recyclable": [remove_dash_from_class_name(item) for item in recyclable_items],
            "non_recyclable": [remove_dash_from_class_name(item) for item in non_recyclable_items],
            "hazardous": [remove_dash_from_class_name(item) for item in hazardous_items]
        }
        
        plotted_image = draw_boxes_on_image(image_np, boxes, confidences, class_ids)
        _, buffer = cv2.imencode(".jpg", plotted_image)
        encoded_image = base64.b64encode(buffer).decode("utf-8")
        
        return {"results": result_dict, "image": encoded_image}
    except Exception as e:
        print(f"Error selama deteksi gambar: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video_feed")
async def video_feed():
    global latest_webcam_results
    webcam_stop_event.clear()
    print("Backend: Webcam stream started, stop event cleared.")
    
    async def generate():
        cap = cv2.VideoCapture(settings.WEBCAM_PATH)
        if not cap.isOpened():
            print("Backend: Error: Could not open webcam.")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes() + b'\r\n')
            return
        
        print("Backend: Webcam opened successfully. Streaming frames...")
        
        prev_frame_time = 0
        new_frame_time = 0

        try:
            while True:
                if webcam_stop_event.is_set():
                    print("Backend: Webcam stop event detected. Breaking loop.")
                    break
                
                success, frame = cap.read()
                if not success:
                    print("Backend: Error: Failed to read frame from webcam. Breaking loop.")
                    break
                
                # Calculate FPS
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                fps_text = f"FPS: {int(fps)}"

                original_shape = frame.shape[:2]
                
                input_image = preprocess_image_for_rknn(frame)
                outputs = rknn.inference(inputs=[input_image])
                boxes, confidences, class_ids, _ = postprocess_yolov8_rknn_output(outputs, original_shape)
                
                detected_items = [settings.ALL_CLASSES[cls_id] for cls_id in class_ids if cls_id < len(settings.ALL_CLASSES)]
                recyclable_items, non_recyclable_items, hazardous_items = classify_waste_type(detected_items)
                
                latest_webcam_results["recyclable"] = [remove_dash_from_class_name(item) for item in recyclable_items]
                latest_webcam_results["non_recyclable"] = [remove_dash_from_class_name(item) for item in non_recyclable_items]
                latest_webcam_results["hazardous"] = [remove_dash_from_class_name(item) for item in hazardous_items]
                
                # Pass fps_text to draw_boxes_on_image
                plotted_image = draw_boxes_on_image(frame, boxes, confidences, class_ids, fps_text)
                _, buffer = cv2.imencode(".jpg", plotted_image)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                await asyncio.sleep(0.001) # Keep this sleep very small to allow for high FPS
        finally:
            if cap.isOpened():
                cap.release()
                print("Backend: Webcam released.")
            else:
                print("Backend: Webcam was not opened, nothing to release.")
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace;boundary=frame")

@app.post("/stop_webcam_backend")
async def stop_webcam_backend():
    webcam_stop_event.set()
    print("Backend: Received stop signal from frontend. Event set.")
    return JSONResponse(content={"message": "Webcam stop signal sent."})

@app.get("/webcam_classification")
async def get_webcam_classification():
    return JSONResponse(content=latest_webcam_results)

@app.on_event("shutdown")
def cleanup():
    rknn.release()
    print("Backend: RKNN model released.")
