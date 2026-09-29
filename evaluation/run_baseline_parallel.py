import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,time,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
from evaluation import baseline_detector as d
cv2.setNumThreads(1)
print('cpu',os.cpu_count(),flush=True)
templates=d.load_templates(Path('pattern_matches'),(.9,1.,1.1))
def score(frame):
 feat=d.preprocess(frame)
 return {name:{state:d.best_match(feat,v).score for state,v in ss.items()} for name,ss in templates.items()}
cap=cv2.VideoCapture('video/original.mp4');scores={name:{s:[] for s in d.STATES[1:]} for name in templates};start=time.time();count=0
with ThreadPoolExecutor(max_workers=8) as pool:
 while True:
  batch=[]
  for _ in range(32):
   ok,f=cap.read()
   if not ok:break
   batch.append(f)
  if not batch:break
  for row in pool.map(score,batch):
   for name,ss in row.items():
    for state,v in ss.items():scores[name][state].append(v)
   count+=1
  print(count,round(time.time()-start,1),flush=True)
states={name:d.decode_states(ss,.72,5) for name,ss in scores.items()}
Path('evaluation/baseline_full.json').write_text(json.dumps({'scores':scores,'states':states,'seconds':time.time()-start}))
