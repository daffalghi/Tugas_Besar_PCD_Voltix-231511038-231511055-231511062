from pathlib import Path
import sys

file_path = Path(__file__).resolve()
root_path = file_path.parent
if root_path not in sys.path:
    sys.path.append(str(root_path))
ROOT = root_path.relative_to(Path.cwd())

# ML Model config
MODEL_DIR = ROOT / 'weights'

# GUNAKAN MODEL .RKNN JIKA INGIN MENGGUNAKAN PERANGKAT 
DETECTION_MODEL = MODEL_DIR / 'best-rk3588.rknn'

# GUNAKAN MODEL .PT (PYTORCH) JIKA INGIN MENGGUNAKAN PC.
# DETECTION_MODEL = MODEL_DIR / 'best.pt'

MODEL_INPUT_WIDTH = 640
MODEL_INPUT_HEIGHT = 640
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45

# Class names (sesuai urutan saat training)
ALL_CLASSES = [
    'battery', 'can', 'cardboard_bowl', 'cardboard_box', 'chemical_plastic_bottle',
    'chemical_plastic_gallon', 'chemical_spray_can', 'light_bulb', 'paint_bucket',
    'plastic_bag', 'plastic_bottle', 'plastic_bottle_cap', 'plastic_box',
    'plastic_cultery', 'plastic_cup', 'plastic_cup_lid', 'reuseable_paper',
    'scrap_paper', 'scrap_plastic', 'snack_bag', 'stick', 'straw'
]

# Webcam
WEBCAM_PATH = 0

# Waste type classification
RECYCLABLE = ['cardboard_box', 'can', 'plastic_bottle_cap', 'plastic_bottle', 'reuseable_paper']
NON_RECYCLABLE = [
    'plastic_bag', 'scrap_paper', 'stick', 'plastic_cup', 'snack_bag',
    'plastic_box', 'straw', 'plastic_cup_lid', 'scrap_plastic',
    'cardboard_bowl', 'plastic_cultery'
]
HAZARDOUS = [
    'battery', 'chemical_spray_can', 'chemical_plastic_bottle',
    'chemical_plastic_gallon', 'light_bulb', 'paint_bucket'
]
