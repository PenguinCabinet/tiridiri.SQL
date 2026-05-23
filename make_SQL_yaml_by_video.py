import cv2
import numpy as np
from pathlib import Path
import yaml
import subprocess

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


threshold = 0.7

frame_i=0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    SQL_template="SELECT * FROM characters WHERE {};"

    show_characters=[]

    for name,status_templates in character_templates.items():
        for status,img_templates in status_templates.items():
            flag=False
            for template in img_templates:
                h, w = template.shape[:2]
                result = cv2.matchTemplate(
                    frame,
                    template,
                    cv2.TM_CCOEFF_NORMED
                )

                locations = np.where(result >= threshold)

                for pt in list(zip(*locations[::-1]))[:1]:
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
                if len(list(zip(*locations[::-1])))>0:
                    flag=True
                    break

            if flag:
                show_characters.append({"name":name,"status":status})
                    

    SQL=SQL_template.format(
        "OR".join(
        ["(name = '{}' AND status = '{}')".format(
            e["name"],
            e["status"],
        ) for e in show_characters]
        if len(show_characters) > 0 
        else
        ["1 = 0"]
    ))

    output_SQL_yaml["body"].append({
        "frame":frame_i,
        "SQL":SQL,
    })

    #cv2.imshow("match", frame)

    opencv_writer.write(frame)

    #if cv2.waitKey(1) == ord("q"):
    #    break

    frame_i+=1

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

