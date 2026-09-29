import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,json,time
from pathlib import Path
from PIL import Image,ImageDraw
import make_SQL_yaml_by_video as d
cv2.setNumThreads(4)
root=Path('evaluation');cap=cv2.VideoCapture('video/original.mp4')
frames=[0,30,60,120,360,600,720,840,960,1200,1320,1560,1800,1920,2040,2160,2280,2400,2520,2640,2760,2880,3000,3120,3360,3480]
templates=d.load_templates(Path('pattern_matches'),(.9,1.,1.1))
result={}
for n in frames:
 cap.set(cv2.CAP_PROP_POS_FRAMES,n);ok,f=cap.read(); features=d.preprocess(f)
 result[n]={name:{s:d.best_match(features,v).__dict__ for s,v in statuses.items()} for name,statuses in templates.items()}
 cv2.imwrite(str(root/f'frame_{n:04}.jpg'),f)
 print(n,flush=True)
(root/'baseline_samples.json').write_text(json.dumps(result,indent=2))
print([(str(p),d.read_image(p).shape[:2]) for p in Path('pattern_matches').glob('*/*/*')])
