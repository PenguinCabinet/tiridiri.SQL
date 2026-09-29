"""Check every generated SQL statement/video frame and render review artifacts."""
import csv
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

from make_SQL_yaml_by_video import load_templates, preprocess, best_match


def main():
    root = Path('evaluation')
    before = json.loads((root/'baseline_full.json').read_text())
    after = json.loads((root/'improved_full.json').read_text())
    document = yaml.safe_load(Path('SQL.yaml').read_text(encoding='utf-8'))
    count = document['meta']['frames']
    assert count == 3573
    connection = sqlite3.connect('file:database.db?mode=ro', uri=True)
    queries = set(row['SQL'] for row in document['body'])
    for query in queries:
        connection.execute(query).fetchall()
    connection.close()
    cap = cv2.VideoCapture('video/OpenCV_processing_process.mp4')
    decoded = 0
    selected = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if decoded in (15, 30, 60, 1967, 2143, 2813, 2880, 3533, 3572):
            selected[decoded] = frame
            cv2.imwrite(str(root/f'output_{decoded:04}.jpg'), frame)
        decoded += 1
    cap.release()
    assert decoded == count
    with Path('video/recognition.csv').open(encoding='utf-8') as stream:
        log = list(csv.DictReader(stream))
    assert len(log) == count*19
    assert all(row['state'] == after['states'][row['character']][int(row['frame'])] for row in log)
    # Source frames plus actual decoded output frames, not just proposed overlays.
    templates = load_templates(Path('pattern_matches'), (.9,1.,1.1))
    source = cv2.VideoCapture('video/original.mp4')
    comparison = Image.new('RGB', (1920, 3*450), 'white')
    draw = ImageDraw.Draw(comparison)
    for index, n in enumerate((1967, 2143, 2880)):
        source.set(cv2.CAP_PROP_POS_FRAMES,n)
        ok, frame = source.read()
        assert ok
        canvas = np.full((420,960,3),22,np.uint8)
        canvas[:360,:640] = frame
        features = preprocess(frame)
        for row, name in enumerate(before['states']):
            state = before['states'][name][n]
            color = (80,230,100) if state=='real' else (60,180,255) if state=='portrait' else (150,150,150)
            cv2.putText(canvas, f'{name:11} {state}', (648,42+row*19), cv2.FONT_HERSHEY_SIMPLEX,.38,color,1)
            if state != 'absent':
                m = best_match(features,templates[name][state])
                x,y = m.point; w,h = m.size
                cv2.rectangle(canvas,(x,y),(x+w,y+h),color,1)
        comparison.paste(Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)),(0,index*450+30))
        comparison.paste(Image.fromarray(cv2.cvtColor(selected[n],cv2.COLOR_BGR2RGB)),(960,index*450+30))
        draw.text((8,index*450+8),f'BEFORE / frame {n} / {n/30:.3f}s', fill='black')
        draw.text((968,index*450+8),'AFTER / decoded output video',fill='black')
    comparison.save(root/'comparison.jpg', quality=92)
    source.release()
    colors = {'absent':(40,40,40),'real':(70,185,100),'portrait':(235,150,55)}
    chart = Image.new('RGB',(1360,880),'white'); draw = ImageDraw.Draw(chart)
    for block, (title,data) in enumerate([('BEFORE',before),('AFTER',after)]):
        ybase=block*440
        draw.text((8,ybase+8),title+'  green=real, orange=portrait, dark=absent',fill='black')
        for row,(name,values) in enumerate(data['states'].items()):
            y=ybase+35+row*20
            draw.text((5,y),name,fill='black')
            for x in range(1200):
                draw.line((150+x,y,150+x,y+14),fill=colors[values[min(count-1,int(x*count/1200))]])
        for sec in range(0,120,10):
            draw.text((150+int(sec/119.1*1200),ybase+420),f'{sec}s',fill='black')
    chart.save(root/'state_timeline.png')
    short_runs=[]
    for name,values in after['states'].items():
        start=0
        for end in range(1,count+1):
            if end==count or values[end]!=values[start]:
                if end-start<=2:
                    short_runs.append({'character':name,'start':start,'end':end-1,'state':values[start]})
                start=end
    report={'decoded_video_frames':decoded,'csv_rows':len(log),'distinct_sql_statements_executed':len(queries),
            'short_runs_at_most_2_frames':short_runs,
            'registration_fallback_frames':sum(m['registration_inliers']==0 for m in after['metadata'])}
    (root/'output_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({**report, 'short_runs_at_most_2_frames':len(short_runs)},indent=2))


if __name__ == '__main__':
    main()
