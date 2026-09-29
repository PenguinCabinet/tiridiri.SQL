import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,json
from pathlib import Path
from layout_detector import LayoutDetector
cv2.setNumThreads(2);d=LayoutDetector(Path('recognition_profile'));out={}
for p in sorted(Path('evaluation').glob('frame_*.jpg')):
 n=int(p.stem.split('_')[1]);m,meta=d.score(cv2.imread(str(p)))
 out[n]={'matches':{name:{s:v.__dict__ for s,v in ss.items()} for name,ss in m.items()},**meta}
 print(n,meta['registration_inliers'],' '.join(name+':'+max(ss,key=lambda s:ss[s].score) for name,ss in m.items() if max(x.score for x in ss.values())>.72),flush=True)
Path('evaluation/final_samples.json').write_text(json.dumps(out,indent=2))
