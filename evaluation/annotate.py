import json
from pathlib import Path
names=sorted(json.load(open('recognition_profile/profile.json'))['characters'])
def names_for(s):
 return ['teacher' if c=='T' else 'studentMain' if c=='X' else 'student'+c for c in s.split()]
rows=[]
def add(frames,default='real',portrait='',real='',absent='',split='development',note=''):
 for f in frames:
  states=dict.fromkeys(names,default)
  for state,letters in [('portrait',portrait),('real',real),('absent',absent)]:
   states.update(dict.fromkeys(names_for(letters),state))
  rows.append({'frame':f,'split':split,'states':states,'note':note})
add([15],portrait='F G',absent='K L Q T')
add([30],portrait='F G H L M N O P',absent='K Q T')
add([45],default='portrait',real='A B C D E T X')
add([60,75,90,360,600,1320,1560,1800,1830,1860,1890,3000],default='portrait',real='X')
add([720,1200,2760,2880])
add([960],portrait='T')
add([1950,1980,2010,2040,2070],absent='F K L Q T',note='Partially cropped faces A-E remain visible; off-screen faces are absent.')
add([2100],default='absent',note='Near-black transition, source maximum pixel value 4.')
add([2160],portrait='T A E F I K')
add([2280],portrait='E F G I J K M')
add([2400],portrait='T E F G J K M O')
add([2520],portrait='B D E G H M')
add([2640],portrait='T D E H L P')
add([3360,3480],default='portrait')
# These frames were selected and visually annotated after freezing iteration 3.
add([165,487,1517,1777,1903,2813,2917],default='portrait',real='X',split='validation')
add([697,1033],split='validation')
add([1255],portrait='F',split='validation')
add([1967,2027],absent='F K L Q T',split='validation')
add([2143],portrait='C D F G I J K',split='validation')
add([2303],portrait='T A B C F G J K M',split='validation')
add([2377],portrait='T A B C D I J',split='validation')
add([2451],portrait='B C D G H I',split='validation')
add([2599],portrait='D E F G H J K L',split='validation')
add([2673],portrait='T E F G J K M',split='validation')
add([3203,3457,3533],default='portrait',split='validation')
add([3572],default='absent',split='validation')
Path('evaluation/labels.json').write_text(json.dumps({'annotation_method':'Manual visual inspection of original source frames, including partial faces. Bodies without visible faces are absent. Labels are not generated from detector predictions.','template_frames_excluded':[0,120,840,1920,3120],'frames':sorted(rows,key=lambda r:r['frame'])},indent=2),encoding='utf-8')
print(len(rows),len(rows)*19)
