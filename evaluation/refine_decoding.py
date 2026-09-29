"""Re-decode the recorded full run, recomputing boxes only for changed states.

Run from the repository root. The CSV and JSON must belong to the same run.
This avoids repeating visual scoring when only the temporal penalty changes.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import yaml
from layout_detector import LayoutDetector
from make_SQL_yaml_by_video import Match, decode_states, sql_for, write_debug_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--switch-cost', type=float, default=1)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    path = Path('evaluation/improved_full.json')
    data = json.loads(path.read_text())
    count = len(next(iter(data['states'].values())))
    states = {n:decode_states(s,.72,args.switch_cost) for n,s in data['scores'].items()}
    matches = [{n:{s:Match(data['scores'][n][s][i],(0,0),(0,0)) for s in ('real','portrait')}
                for n in states} for i in range(count)]
    with Path('video/recognition.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows)==count*len(states)
    for row in rows:
        i=int(row['frame']);n=row['character'];s=row['state']
        assert data['states'][n][i]==s, 'CSV and score cache differ'
        if s!='absent':
            matches[i][n][s]=Match(data['scores'][n][s][i],(int(row['x']),int(row['y'])),
                                   (int(row['width']),int(row['height'])))
    cap=cv2.VideoCapture('video/original.mp4');fps=cap.get(cv2.CAP_PROP_FPS)
    detector=LayoutDetector(Path('recognition_profile'))
    recomputed=0
    for i in range(count):
        if not any(states[n][i]!='absent' and states[n][i]!=data['states'][n][i] for n in states):
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,frame=cap.read();assert ok
        found,_=detector.score(frame)
        for n in states:
            for s,m in found[n].items():
                assert abs(m.score-data['scores'][n][s][i])<1e-4, f'Visual scoring changed: {i} {n} {s}'
        matches[i]=found;recomputed+=1
    cap.release()
    data['states']=states
    data['configuration']={'threshold':.72,'switch_cost':args.switch_cost,'detector':'layout'}
    document={'meta':{'fps':fps,'frames':count,**data['configuration']},
              'body':[{'frame':i,'SQL':sql_for(states,i)} for i in range(count)]}
    sql_temp = Path('SQL.yaml.new')
    sql_temp.write_text(yaml.safe_dump(document,allow_unicode=True,sort_keys=False),encoding='utf-8')
    for row in rows:
        i=int(row['frame']);n=row['character'];s=states[n][i]
        row['state']=s
        m=matches[i][n].get(s)
        for key,value in zip(('x','y','width','height'),m.point+m.size if m else ('','','','')):
            row[key]=value
    csv_temp = Path('video/recognition.csv.new')
    with csv_temp.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temp=Path('video/OpenCV_processing_process_temp.mp4')
    print(f'Checked {recomputed} changed frames; rendering video.',flush=True)
    write_debug_video(temp,Path('video/original.mp4'),states,matches,fps)
    subprocess.run(['ffmpeg','-loglevel','error','-y','-i',str(temp),'-i','video/original.mp4',
                    '-map','0:v:0','-map','1:a:0?','-c:v','libx264','-crf','18','-preset','fast',
                    '-pix_fmt','yuv420p','-movflags','+faststart','-c:a','copy',
                    'video/OpenCV_processing_process_final_temp.mp4'],check=True)
    temp.unlink()
    sql_temp.replace('SQL.yaml')
    csv_temp.replace('video/recognition.csv')
    Path('video/OpenCV_processing_process_final_temp.mp4').replace('video/OpenCV_processing_process.mp4')
    path.write_text(json.dumps(data))
    print(f'Re-decoded {count} frames; rechecked visual scores/boxes on {recomputed} changed frames.')


if __name__=='__main__':
    main()
