from ultralytics import YOLO

def main():
    # Tentukan path ke file data.yaml
    data_yaml = r"C:/Users/daffa/Desktop/YOLO8/waste-detection/dataset/data.yaml"

    # Muat model YOLOv8n (pre-trained)
    model = YOLO("yolov8n.pt")  # Menggunakan pre-trained weights

    # Training
    print("Memulai training...")
    train_results = model.train(
        data=data_yaml,        # Path ke data.yaml
        epochs=50,             # Jumlah epoch
        imgsz=640,             # Ukuran gambar
        batch=16,              # Ukuran batch
        device=0,              # Gunakan GPU (ganti ke "cpu" jika tidak ada GPU)
        name="waste_detection" # Nama folder hasil training
    )

    # Validasi
    print("Memulai validasi...")
    val_results = model.val(
        data=data_yaml,
        imgsz=640,
        batch=16,
        device=0
    )

    # Prediksi pada data test
    print("Memulai prediksi...")
    predict_results = model.predict(
        source=r"C:/Users/daffa/Desktop/YOLO8/waste-detection/dataset/test/images",
        save=True,              # Simpan hasil prediksi
        imgsz=640,
        device=0
    )

    print("Proses selesai!")
    print(f"Hasil training disimpan di: runs/train/waste_detection")
    print(f"Model terbaik: runs/train/waste_detection/weights/best.pt")
    print(f"Hasil prediksi disimpan di: runs/predict/")

if __name__ == '__main__':
    main()