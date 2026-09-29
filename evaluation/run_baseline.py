import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation import baseline_detector as d
import cv2
cv2.setNumThreads(2)
start=time.time();cap=cv2.VideoCapture('video/original.mp4')
scores,matches=d.score_video(cap,d.load_templates(Path('pattern_matches'),(.9,1.,1.1)))
states={name:d.decode_states(s,.72,5) for name,s in scores.items()}
Path('evaluation/baseline_full.json').write_text(json.dumps({'states':states,'scores':scores,'seconds':time.time()-start}))
print('DONE',time.time()-start,flush=True)
