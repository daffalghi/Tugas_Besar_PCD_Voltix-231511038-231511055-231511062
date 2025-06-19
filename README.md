# 🗑️ Sistem Pemilahan Sampah Cerdas

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-green.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-red.svg)](https://fastapi.tiangolo.com/)

Sistem deteksi sampah berbasis AI menggunakan model YOLOv8 untuk mengidentifikasi dan mengklasifikasikan sampah menjadi kategori **Dapat Didaur Ulang**, **Tidak Dapat Didaur Ulang**, dan **Berbahaya** secara real-time melalui webcam.

## 📋 Daftar Isi

- [Fitur](#-fitur)
- [Dataset](#-dataset)
- [Instalasi](#-instalasi)
- [Penggunaan](#-penggunaan)
- [Struktur Proyek](#-struktur-proyek)
- [Klasifikasi Sampah](#-klasifikasi-sampah)
- [API Endpoints](#-api-endpoints)
- [Screenshot](#-screenshot)
- [Kontribusi](#-kontribusi)
- [Lisensi](#-lisensi)
- [Referensi](#-referensi)

## ✨ Fitur

- 🎯 **Deteksi Real-time**: Deteksi sampah langsung dari webcam
- 🔄 **Klasifikasi Otomatis**: Mengelompokkan sampah ke dalam 3 kategori utama
- 🚀 **API RESTful**: Interface API menggunakan FastAPI
- 📊 **Model YOLOv8**: Menggunakan state-of-the-art object detection
- 💻 **Web Interface**: Antarmuka web yang user-friendly
- ⚡ **Real-time Processing**: Pemrosesan video stream secara real-time

## 📊 Dataset

Dataset yang digunakan untuk melatih model tersedia di:
```
https://universe.roboflow.com/ai-project-i3wje/waste-detection-vqkjo/model/3
```

Dataset ini berisi berbagai jenis sampah yang telah dianotasi untuk training model deteksi objek.

## 🚀 Instalasi

### Prasyarat
- Python 3.8 atau lebih tinggi
- Webcam (untuk deteksi real-time)
- Git

### Langkah Instalasi

1. **Clone Repository**
   ```bash
   git clone https://github.com/daffalghi/wastedetection_yolov8.git
   cd waste-detection
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan Aplikasi**
   ```bash
   uvicorn main:app --reload
   ```

4. **Akses Aplikasi**
   
   Buka browser dan navigasikan ke:
   ```
   http://127.0.0.1:8000
   ```

### Langkah Konversi Model PT ke RKNN
Install Requirement baru
```bash
   pip install rknn-toolkit2
   ```
1. **Jalankan Program Konversi**
   ```bash
   python convert_pt_to_rknn.py
   ```
   Pastikan sudah memiliki model best.pt yang tersimpan pada folder weight.

2. **Pindahkan Seluruh File ke Orange Pi 5 Pro**
3. **Jalankan Aplikasi pada Orange Pi 5 Pro**
   Pastikan dependensi dan requirement sudah terinstall pada Orange Pi 5 Pro dengan menjalankan
   ```bash
   pip install -r requirements.txt
   ```
4. **Jalankan Aplikasi main1.py**
   ```bash
   uvicorn main1:app --reload
   ```

5. **Akses Aplikasi**
   
   Buka browser dan navigasikan ke:
   ```
   http://127.0.0.1:8000
   ```
   
## 📖 Penggunaan

1. Pastikan webcam terhubung dengan komputer
2. Jalankan aplikasi menggunakan perintah di atas
3. Buka browser dan akses URL yang disediakan
4. Arahkan objek sampah ke webcam
5. Sistem akan otomatis mendeteksi dan mengklasifikasikan sampah

## 📁 Struktur Proyek

```
waste-detection/
│
├── main.py              # Aplikasi utama FastAPI
├── helper.py            # Fungsi pembantu untuk deteksi YOLO
├── settings.py          # Konfigurasi dan pengaturan
├── train.py            # Script untuk training model
├── requirements.txt    # Dependencies Python
├── README.md          # Dokumentasi proyek
│
├── models/            # Model YOLOv8 yang telah dilatih
├── static/           # File statis (CSS, JS, gambar)
└── templates/        # Template HTML
```

### Deskripsi File Utama

| File | Deskripsi |
|------|-----------|
| `main.py` | Berkas aplikasi utama yang berisi kode FastAPI untuk menjalankan web server dan API endpoints |
| `helper.py` | Fungsi pembantu untuk deteksi sampah menggunakan model YOLO, termasuk preprocessing dan postprocessing |
| `settings.py` | Pengaturan konfigurasi aplikasi, jalur model YOLO, dan definisi jenis sampah |
| `train.py` | Script untuk melatih ulang model dengan dataset baru |

## 🗂️ Klasifikasi Sampah

Sistem mengklasifikasikan sampah ke dalam 3 kategori utama:

### ♻️ DAPAT DIDAUR ULANG (RECYCLABLE)
```python
['cardboard_box', 'can', 'plastic_bottle_cap', 'plastic_bottle', 'reuseable_paper']
```

### 🚫 TIDAK DAPAT DIDAUR ULANG (NON_RECYCLABLE)
```python
['plastic_bag', 'scrap_paper', 'stick', 'plastic_cup', 'snack_bag', 
 'plastic_box', 'straw', 'plastic_cup_lid', 'scrap_plastic', 
 'cardboard_bowl', 'plastic_cultery']
```

### ⚠️ BERBAHAYA (HAZARDOUS)
```python
['battery', 'chemical_spray_can', 'chemical_plastic_bottle', 
 'chemical_plastic_gallon', 'light_bulb', 'paint_bucket']
```

## 🔌 API Endpoints

| Endpoint | Method | Deskripsi |
|----------|--------|-----------|
| `/` | GET | Halaman utama aplikasi (HTML interface) |
| `/detect` | POST | Upload dan deteksi sampah dari file gambar |
| `/video_feed` | GET | Stream video real-time dari webcam dengan deteksi |
| `/stop_webcam_backend` | POST | Menghentikan stream webcam dari backend |
| `/webcam_classification` | GET | Mendapatkan hasil klasifikasi terbaru dari webcam |
| `/docs` | GET | Dokumentasi API otomatis (Swagger UI) |

### Detail API Endpoints

#### POST `/detect`
- **Fungsi**: Upload gambar untuk deteksi sampah
- **Input**: File gambar (multipart/form-data)
- **Output**: JSON dengan hasil klasifikasi dan gambar yang telah dianotasi
- **Response Format**:
  ```json
  {
    "results": {
      "recyclable": ["plastic bottle", "cardboard box"],
      "non_recyclable": ["plastic bag", "straw"],
      "hazardous": ["battery"]
    },
    "image": "base64_encoded_annotated_image"
  }
  ```

#### GET `/video_feed`
- **Fungsi**: Stream video real-time dari webcam
- **Output**: Multipart stream (MJPEG format)
- **Content-Type**: `multipart/x-mixed-replace;boundary=frame`

#### GET `/webcam_classification`
- **Fungsi**: Mendapatkan hasil klasifikasi terbaru dari deteksi webcam
- **Output**: JSON dengan klasifikasi sampah yang terdeteksi
- **Response Format**:
  ```json
  {
    "recyclable": ["can", "plastic bottle"],
    "non_recyclable": ["plastic cup"],
    "hazardous": []
  }
  ```

#### POST `/stop_webcam_backend`
- **Fungsi**: Menghentikan stream webcam dari sisi backend
- **Output**: Konfirmasi penghentian stream
- **Response**: `{"message": "Webcam stop signal sent."}`

## 📸 Screenshot

<!-- Tambahkan screenshot aplikasi di sini -->
*Screenshot akan ditambahkan setelah aplikasi berjalan*

## 🤝 Kontribusi

Kontribusi sangat diterima! Untuk berkontribusi:

1. Fork repository ini
2. Buat branch fitur baru (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE) - lihat file LICENSE untuk detail lebih lanjut.

## 🔗 Referensi

- [Dokumentasi FastAPI](https://fastapi.tiangolo.com/)
- [Dokumentasi YOLOv8](https://docs.ultralytics.com/)
- [Roboflow Dataset](https://universe.roboflow.com/ai-project-i3wje/waste-detection-vqkjo/model/3)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

---

## 👨‍💻 Developer

Dikembangkan dengan ❤️ untuk lingkungan yang lebih bersih

**Hubungi Developer:**
- GitHub: [@daffalghi](https://github.com/daffalghi)

---

*Jika Anda menemukan bug atau memiliki saran, silakan buat issue di repository ini.*
