import cv2,json
import numpy as np
from pathlib import Path
import sys, hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from make_SQL_yaml_by_video import read_image
out=Path('recognition_profile');out.mkdir(exist_ok=True)
cap=cv2.VideoCapture('video/original.mp4')
def frame(n):
 cap.set(1,n);ok,f=cap.read();assert ok;return f
r=json.load(open('evaluation/baseline_samples.json'))
real=frame(840);portrait=frame(120);ending=frame(3120);opening=frame(0)
cv2.imwrite(str(out/'solo.png'),portrait);cv2.imwrite(str(out/'group.png'),real);cv2.imwrite(str(out/'opening.png'),opening)
config={'source_size':[640,360],'reference_frames':{'group':840,'opening':0,'portraits':120,'main_portrait':3120},'opening_transform':[[8.214296,-.107224,-2285.562],[.107224,8.214296,-1332.757]],'characters':{}}
for name in sorted(r['840']):
 config['characters'][name]={}
 for state,n,f in [('real','840',real),('portrait','3120' if name=='studentMain' else '120',ending if name=='studentMain' else portrait)]:
  m=r[n][name][state];x,y=m['point'];w,h=m['size']
  file=f'{name}_{state}.png';cv2.imwrite(str(out/file),f[y:y+h,x:x+w]);config['characters'][name][state]={'box':[x,y,w,h],'images':[file]}
# Closed eyes and the solo pose are distinct appearances, stored separately.
H=np.array([[2.2290575,-.000044675,-393.03233],[.000044675,2.2290575,-220.93516]],np.float32)
closed=cv2.warpAffine(frame(1920),cv2.invertAffineTransform(H),(640,360))
for key,f in [('closed',closed),('solo',portrait)]:
 x,y,w,h=config['characters']['studentMain']['real']['box'];file=f'studentMain_real_{key}.png'
 cv2.imwrite(str(out/file),f[y:y+h,x:x+w]);config['characters']['studentMain']['real']['images'].append(file)
config['reference_frames'].update({'solo':120,'closed_eyes':1920})
config['source_sha256']=hashlib.sha256(Path('video/original.mp4').read_bytes()).hexdigest()
(out/'profile.json').write_text(json.dumps(config,indent=2))
