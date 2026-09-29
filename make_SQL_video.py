import subprocess
import cv2
import numpy as np
import yaml
from pathlib import Path
import hashlib
import os

with open("SQL.yaml", "r", encoding="utf-8") as f:
    SQL = yaml.safe_load(f)

cap = cv2.VideoCapture("video/original.mp4")

SQL_video_width=3840
SQL_video_height=2160

SQL_video_writer = cv2.VideoWriter(
    "video/SQL_temp.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    cap.get(cv2.CAP_PROP_FPS),
    (
        SQL_video_width,
        SQL_video_height
    )
)

os.makedirs(
    "temp_silicon_images",exist_ok=True
)
previous_digest = None
silicon_img = None

for frame_i,elem in enumerate(SQL["body"]):
    if frame_i%100==0:
        print(frame_i)

    digest = hashlib.sha256(elem["SQL"].encode()).digest().hex()
    image_path = Path("temp_silicon_images") / f"{digest}.png"
    if not image_path.is_file():
        SQL_run_output = subprocess.run([
            "sqlite3",
            "database.db",
            "-header", "-column",
            elem["SQL"].replace("\n","")
        ], 
        shell=True, capture_output=True, text=True,
        encoding='cp932',
        timeout=10).stdout

        silicon_cmd = [
            "silicon",
            "--font","Hack=45",
            "--language", "sql",
            "--output", str(image_path)
        ]
        output_text="{}\n{}".format(
            elem["SQL"],
            SQL_run_output,
        )

        subprocess.run(
            silicon_cmd, 
            input=output_text.encode('utf-8'), 
            capture_output=True, 
            text=False, # バイナリとして受け取るためFalse
            check=True
        )

    if digest != previous_digest:
        silicon_img = cv2.imread(str(image_path))
        if silicon_img is None:
            raise RuntimeError(f"Could not read rendered SQL image: {image_path}")
        previous_digest = digest

    silicon_img_h, silicon_img_w = silicon_img.shape[:2]


    x = (SQL_video_width - silicon_img_w) // 2
    y = (SQL_video_height - silicon_img_h) // 2

    # Always start with a clean background; cached images must not leave pixels
    # from the previous frame when the SQL panel becomes smaller.
    frame_img = np.full(
        (SQL_video_height, SQL_video_width, 3),
        (255, 170, 170),
        dtype=np.uint8,
    )
    frame_img[
        y:y+silicon_img_h,
        x:x+silicon_img_w
    ] = silicon_img

    SQL_video_writer.write(frame_img)

cap.release()
SQL_video_writer.release()

ffmpeg_cmd = [
    "ffmpeg",
    "-i", "video/SQL_temp.mp4",
    "-i", "video/original.mp4",
    "-c:v", "copy",
    "-c:a", "copy",
    "-map", "0:v:0",
    "-map", "1:a:0",
    "video/SQL_new.mp4",
    "-y"
]

subprocess.run(ffmpeg_cmd, check=True)
os.replace("video/SQL_new.mp4", "video/SQL.mp4")

if Path("video/SQL_temp.mp4").is_file():
    Path("video/SQL_temp.mp4").unlink()


