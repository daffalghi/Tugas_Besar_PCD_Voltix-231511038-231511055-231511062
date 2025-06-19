import os
from pathlib import Path

# Tentukan direktori dataset di Orange Pi
dataset_dir = Path("dataset")  # Sesuaikan dengan jalur di Orange Pi
output_file = dataset_dir / "dataset.txt"

# Folder yang akan diproses
folders = ["test"]

# Ekstensi file gambar yang valid
valid_extensions = {".jpg", ".jpeg", ".png"}

# Kumpulkan semua file gambar
image_files = []
for folder in folders:
    images_dir = dataset_dir / folder / "images"
    if images_dir.exists():
        image_files.extend([f for f in images_dir.iterdir() if f.suffix.lower() in valid_extensions])
    else:
        print(f"Direktori tidak ditemukan: {images_dir}")

# Urutkan file untuk konsistensi
image_files = sorted(image_files)

# Buat file dataset.txt
if image_files:
    with open(output_file, "w") as f:
        for image_path in image_files:
            f.write(f"{image_path.resolve()}\n")
    print(f"File dataset.txt dibuat di: {output_file} dengan {len(image_files)} gambar.")
else:
    print("Tidak ada file gambar ditemukan.")
