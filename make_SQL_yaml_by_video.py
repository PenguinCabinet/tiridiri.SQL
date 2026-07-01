import cv2
import numpy as np
from pathlib import Path
import yaml
import subprocess
from concurrent.futures import ThreadPoolExecutor
import copy

cap = cv2.VideoCapture("video/original.mp4")

if not cap.isOpened():
    print("video/original.mp4の動画を開けません")
    exit()

character_templates={}

for p in Path("./pattern_matches").iterdir():
    if p.is_dir():
        character_templates[p.name]={}
        for status in (Path("./pattern_matches")/Path(p.name)).iterdir():
            character_templates[p.name][status.name]=[]
            for image in (Path("./pattern_matches")/Path(p.name)/Path(status.name)).iterdir():
                character_templates[p.name][status.name].append(
                    cv2.imread(
                        (image)
                    )
                )


output_SQL_yaml={
    "meta":{},
    "body":[]
}

opencv_writer = cv2.VideoWriter(
    "video/OpenCV_processing_process_temp.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    cap.get(cv2.CAP_PROP_FPS),
    (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
)

def my_join(s1,s2,r,arr):
    result=""
    for i in range(len(arr)):
        s=s1
        if i%r==0:
            s+=s2
        result+=arr[i]+(s if i!=len(arr)-1 else "")
    
    return result

threshold = 0.8

def detect_from_frame(args):
    frame_i,frame_original=tuple(args)
    frame=frame_original.copy()

    show_characters=[]

    for name,status_templates in character_templates.items():
        locations_status_dict={}
        max_result_status_dict={}
        for status,img_templates in status_templates.items():
            flag=False
            for template in img_templates:
                h, w = template.shape[:2]
                result = cv2.matchTemplate(
                    frame_original,
                    template,
                    cv2.TM_CCOEFF_NORMED
                )

                locations = np.where(result >= threshold)
                locations_status_dict[status]=list(zip(*locations[::-1]))[:1]
                max_result_status_dict[status]=np.max(result)
                
        if len(max_result_status_dict)==0:
            break
        plausible_status = max(max_result_status_dict, key=max_result_status_dict.get)

        if max_result_status_dict[plausible_status]>=threshold:
            flag=True
            for pt in locations_status_dict[plausible_status]:
                    cv2.putText(
                        frame,
                        "{},{}".format(
                            name,status
                        ),
                        (max(0,pt[0] - 60), pt[1] ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 255, 0),
                        1
                    )
                    cv2.rectangle(
                        frame,
                        pt,
                        (pt[0] + w, pt[1] + h),
                        (0, 255, 0),
                        2
                    )

            if flag:
                show_characters.append({"name":name,"status":plausible_status})
    SQL=SQL_template.format(
        my_join(
        " OR ",
        "\n",
        2,
        ["(name = '{}' AND status = '{}')".format(
            e["name"],
            e["status"],
        ) for e in show_characters]
        if len(show_characters) > 0 
        else
        ["1 = 0"]
    ))

    return {
        "frame":frame_i,
        "SQL":SQL,
    },frame


SQL_template="SELECT * FROM characters WHERE {};"
frame_i=0
frames=[]
while True:
    ret, frame = cap.read()

    if not ret:
        break
    frames.append(frame)
                    
with ThreadPoolExecutor(max_workers=16) as executor:
    result = list(executor.map(detect_from_frame, enumerate(frames)))
    #cv2.imshow("match", frame)
    result.sort(key=lambda v:v[0]["frame"])

    for (out_yaml,frame) in result:
        opencv_writer.write(frame)
        output_SQL_yaml["body"].append(out_yaml)

print(len([e["frame"] for e in output_SQL_yaml["body"]]))

with open("SQL.yaml", "w", encoding="utf-8") as f:
    yaml.dump(output_SQL_yaml, f)

cap.release()
opencv_writer.release()

ffmpeg_cmd = [
    "ffmpeg",
    "-i", "video/OpenCV_processing_process_temp.mp4",
    "-i", "video/original.mp4",
    "-c:v", "copy",
    "-c:a", "copy",
    "-map", "0:v:0",
    "-map", "1:a:0",
    "video/OpenCV_processing_process.mp4",
    "-y"
]

subprocess.run(ffmpeg_cmd)

if Path("video/OpenCV_processing_process_temp.mp4").is_file():
    Path("video/OpenCV_processing_process_temp.mp4").unlink()

cv2.destroyAllWindows()

