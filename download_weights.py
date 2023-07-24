import gdown
import zipfile
from pathlib import Path

Path("./weights").mkdir(parents=True, exist_ok=True)

model_urls = [
    ("yolov5l_full.pt", 'https://drive.google.com/file/d/1TEKuVRz4PGfL4F34OcjFBdaruRjpV9Ev/view?usp=sharing'),
    ("weights_domain.zip", 'https://drive.google.com/file/d/1tQCOKoaFSnusIeFWNH6qQewbz91YrrtJ/view?usp=sharing'),
    ("weights_size.zip", 'https://drive.google.com/file/d/14kX114mz7ZeUPaY-HLSPas7mmZ0fl5kE/view?usp=sharing'),
    ("weights_compression.zip", "https://drive.google.com/file/d/1B_ikby3-Y-h-C1qmD5KKFoGTi8IjM-yE/view?usp=sharing")

]

for name, url in model_urls:
    gdown.download(url, output=f"./weights/{name}", fuzzy=True)

with zipfile.ZipFile('./weights/weights_domain.zip', 'r') as zip_ref:
    zip_ref.extractall('./weights/')

with zipfile.ZipFile('./weights/weights_size.zip', 'r') as zip_ref:
    zip_ref.extractall('./weights/')

with zipfile.ZipFile('./weights/weights_compression.zip', 'r') as zip_ref:
    zip_ref.extractall('./weights/')
