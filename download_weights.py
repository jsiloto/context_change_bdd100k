import gdown
import zipfile
from pathlib import Path

Path("./weights").mkdir(parents=True, exist_ok=True)

model_urls = [
    ("yolov5l_full.pt", 'https://drive.google.com/file/d/1TEKuVRz4PGfL4F34OcjFBdaruRjpV9Ev/view?usp=sharing'),
    ("weights.zip", 'https://drive.google.com/file/d/1tQCOKoaFSnusIeFWNH6qQewbz91YrrtJ/view?usp=sharing')
]

for name, url in model_urls:
    gdown.download(url, output=f"./weights/{name}", fuzzy=True)

with zipfile.ZipFile('./weights/weights.zip', 'r') as zip_ref:
    zip_ref.extractall('./weights/')
