import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,json
import numpy as np
from PIL import Image,ImageDraw
cap=cv2.VideoCapture('video/original.mp4');frames=[0,15,30,45,60,75,90,1800,1830,1860,1890,1920,1950,1980,2010,2040,2070,2100]
out=Image.new('RGB',(1280,5*204),'white');dr=ImageDraw.Draw(out)
for i,n in enumerate(frames):
 cap.set(1,n);_,f=cap.read();cv2.imwrite(f'evaluation/frame_{n:04}.jpg',f)
 out.paste(Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)).resize((320,180)),((i%4)*320,(i//4)*204+24));dr.text(((i%4)*320+3,(i//4)*204+3),str(n),fill='black')
out.save('evaluation/zooms.jpg')
sift=cv2.SIFT_create(nfeatures=1800,contrastThreshold=.02)
ref=cv2.imread('evaluation/frame_0840.jpg');mask=np.zeros(ref.shape[:2],np.uint8);mask[85:330,70:550]=255
k1,d1=sift.detectAndCompute(ref,mask)
for n in [0,30,60,120,600,840,1320,1920,2040,2160,2520,3120]:
 f=cv2.imread(f'evaluation/frame_{n:04}.jpg');k2,d2=sift.detectAndCompute(f,None)
 matches=cv2.BFMatcher().knnMatch(d1,d2,k=2);good=[a for a,b in matches if a.distance<.7*b.distance]
 a=np.float32([k1[m.queryIdx].pt for m in good]);b=np.float32([k2[m.trainIdx].pt for m in good])
 H,inl=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=3)
 print(n,len(good),int(inl.sum()),H.tolist())
