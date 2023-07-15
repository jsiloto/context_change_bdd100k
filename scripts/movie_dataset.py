import os
import shutil
import argparse
from PIL import Image
from tqdm import tqdm
from moviepy.editor import *

parser = argparse.ArgumentParser(description='bdd2coco')
parser.add_argument('--movie_dir', type=str, required=True)
parser.add_argument('--frames', type=int, default=20000)


def convert_mov_to_jpeg(input_dir, input_file, output_dir, num_frames):
    video = VideoFileClip(os.path.join(input_dir, input_file))
    frames = video.iter_frames(fps=4)
    name = input_file.split(".")[0]
    for i, frame in enumerate(frames):
        output_path = f"{output_dir}/{name}_frame_{i:04d}.jpeg"
        image = Image.fromarray(frame).resize((640, 640))
        image.save(output_path)
        if i > num_frames:
            break
    video.reader.close()

def main(cfg):
    movies = os.listdir(cfg.movie_dir)
    movies.sort()
    output_dir = cfg.movie_dir+"_frames"
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    for m in tqdm(movies):
        convert_mov_to_jpeg(cfg.movie_dir, m, output_dir, cfg.frames)

if __name__ == '__main__':
    cfg = parser.parse_args()
    main(cfg)
