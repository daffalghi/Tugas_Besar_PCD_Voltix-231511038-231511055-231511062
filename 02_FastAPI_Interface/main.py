from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import settings
import io
from PIL import Image
import base64
import time # Import the time module
import json
import asyncio

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

model_path = Path(settings.DETECTION_MODEL)
try:
    model = YOLO(model_path)
except Exception as ex:
    raise Exception(f"Unable to load model. Check the specified path: {model_path} - {ex}")

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

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        results = model.predict(image, conf=0.6, verbose=False)
        names = model.names
        detected_items = set([names[int(c)] for c in results[0].boxes.cls])

        recyclable_items, non_recyclable_items, hazardous_items = classify_waste_type(detected_items)
        
        result_dict = {
            "recyclable": [remove_dash_from_class_name(item) for item in recyclable_items],
            "non_recyclable": [remove_dash_from_class_name(item) for item in non_recyclable_items],
            "hazardous": [remove_dash_from_class_name(item) for item in hazardous_items]
        }

        res_plotted = results[0].plot()
        _, buffer = cv2.imencode(".jpg", res_plotted)
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        return {"results": result_dict, "image": encoded_image}
    except Exception as e:
        print(f"Error during image detection: {e}") 
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
        
        # Inisialisasi variabel untuk perhitungan FPS
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
                
                # Hitung FPS
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                fps_text = f"FPS: {int(fps)}"

                results = model.predict(frame, conf=0.6, verbose=False)
                names = model.names
                detected_items = set([names[int(c)] for c in results[0].boxes.cls])

                recyclable_items, non_recyclable_items, hazardous_items = classify_waste_type(detected_items)
                
                latest_webcam_results["recyclable"] = [remove_dash_from_class_name(item) for item in recyclable_items]
                latest_webcam_results["non_recyclable"] = [remove_dash_from_class_name(item) for item in non_recyclable_items]
                latest_webcam_results["hazardous"] = [remove_dash_from_class_name(item) for item in hazardous_items]

                # Dapatkan gambar yang sudah diplot oleh YOLO
                res_plotted = results[0].plot()
                
                # Tambahkan teks FPS ke gambar yang sudah diplot
                cv2.putText(res_plotted, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                _, buffer = cv2.imencode(".jpg", res_plotted)
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