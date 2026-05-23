import subprocess
import cv2
import numpy as np
import yaml
from pathlib import Path

with open("SQL.yaml", "r", encoding="utf-8") as f:
    SQL = yaml.safe_load(f)

cap = cv2.VideoCapture("video/original.mp4")

SQL_video_width=1920
SQL_video_height=1080

SQL_video_writer = cv2.VideoWriter(
    "video/SQL_temp.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    cap.get(cv2.CAP_PROP_FPS),
    (
        SQL_video_width,
        SQL_video_height
    )
)

for frame_i,elem in enumerate(SQL["body"]):
    if frame_i%100==0:
        print(frame_i)
    SQL_run_output = subprocess.run([
        "sqlite3",
        "database.db",
        "\"{}\"".format(elem["SQL"])
    ], 
    shell=True, capture_output=True, text=True,
    encoding='cp932',
    timeout=10).stdout

    silicon_cmd = [
        "silicon",
        "--language", "sql",
        "--output","./temp.png"
    ]
    output_text="> {}\n{}".format(
        elem["SQL"],
        SQL_run_output,
    )

    silicon_res = subprocess.run(
        silicon_cmd, 
        input=output_text.encode('utf-8'), 
        capture_output=True, 
        text=False, # バイナリとして受け取るためFalse
        check=True
    )

    background_color = (255, 170,170)
    frame_img = np.full(
        (SQL_video_height, SQL_video_width, 3),
        background_color,
        dtype=np.uint8
    )

    silicon_img=cv2.imread("./temp.png")
    silicon_img_h, silicon_img_w = silicon_img.shape[:2]


    x = (SQL_video_width - silicon_img_w) // 2
    y = (SQL_video_height - silicon_img_h) // 2

    frame_img[
        y:y+silicon_img_h,
        x:x+silicon_img_w
    ] = silicon_img

    SQL_video_writer.write(frame_img)

    #cv2.imshow("Super Modern Terminal (via silicon)", frame_img)
    #cv2.waitKey(0)

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
    "video/SQL.mp4",
    "-y"
]

subprocess.run(ffmpeg_cmd)

if Path("video/SQL_temp.mp4").is_file():
    Path("video/SQL_temp.mp4").unlink()

cv2.destroyAllWindows()








