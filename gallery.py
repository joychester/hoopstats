import cv2,numpy as np,glob,sys,re
d=sys.argv[1]; pat=sys.argv[2] if len(sys.argv)>2 else 'G'
fs=[f for f in glob.glob(f'{d}/crops/{pat}*.jpg') if re.fullmatch(rf'{pat}\d+\.jpg', f.split('/')[-1])]
fs=sorted(fs,key=lambda f:int(re.findall(r'\d+',f.split('/')[-1])[0]))
def sheet(fs,name):
    W=760; rows=[]
    for f in fs:
        i=cv2.imread(f)[:, :W-90]; r=np.full((170,W,3),255,np.uint8); r[5:165,90:90+i.shape[1]]=i
        cv2.putText(r,f.split('/')[-1][:-4],(5,95),0,1.1,(0,0,0),3); rows.append(r)
    cv2.imwrite(name,np.vstack(rows))
for i in range(0,len(fs),12): sheet(fs[i:i+12],f'{d}/gallery_{i//12}.jpg')
print(len(fs))
