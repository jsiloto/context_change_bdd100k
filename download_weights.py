import gdown
from pathlib import Path

Path("./weights").mkdir(parents=True, exist_ok=True)

model_urls = [
    ("yolov5l_baseline.pt", 'https://drive.google.com/file/d/1TEKuVRz4PGfL4F34OcjFBdaruRjpV9Ev/view?usp=sharing'),
    ("yolov5s_daytime.pt", 'https://drive.google.com/file/d/1oaNW1-7gejJPDuHHrUwF1GlV8WQqse6U/view?usp=sharing'),
    ("yolov5s_city_street.pt", 'https://drive.google.com/file/d/1OsspSWDWNFEzQGo48sLYdLyUtQZjDNzz/view?usp=sharing')
]

for name, url in model_urls:
    gdown.download(url, output=f"./weights/{name}", fuzzy=True)
