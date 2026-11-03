"""Build montages of nano-vs-reference disagreements so they can be checked by eye."""
import os, sys, json, cv2, numpy as np
from ultralytics import YOLO
src=open('code/experiments_cs2.py').read()
env={"np":np,"cv2":cv2}; exec("def iou_matrix"+src.split("def iou_matrix")[1].split("def prf")[0],env)
sf="def sample_frames"+src.split("def sample_frames")[1].split("def predict")[0]; exec(sf,env)
V=[1,2,3,5,7]; nano=YOLO('models/yolov8n.pt'); xl=YOLO('models/yolov8x.pt')
def pred(m,f,s,c):
    b=m.predict(f,imgsz=s,conf=c,classes=V,device='mps',verbose=False)[0].boxes
    return b.xyxy.cpu().numpy(),b.conf.cpu().numpy(),b.cls.cpu().numpy().astype(int)
def montage(items,name):
    rng=np.random.default_rng(1); idx=rng.permutation(len(items))[:40]; tiles=[];meta=[]
    for n,i in enumerate(idx):
        f,b,c=items[i]; x1,y1,x2,y2=map(int,b); w,h=x2-x1,y2-y1; p=max(w,h,40)
        X1,Y1=max(0,x1-p),max(0,y1-p); X2,Y2=min(f.shape[1],x2+p),min(f.shape[0],y2+p)
        t=f[Y1:Y2,X1:X2].copy(); cv2.rectangle(t,(x1-X1,y1-Y1),(x2-X1,y2-Y1),(0,255,255),1)
        t=cv2.resize(t,(230,170),interpolation=cv2.INTER_CUBIC); cv2.rectangle(t,(0,0),(40,22),(0,0,0),-1); cv2.putText(t,str(n+1),(3,17),0,.55,(255,255,255),1); tiles.append(t); meta.append({"id":n+1,"conf":float(c),"w":w,"h":h})
    while len(tiles)%8: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(name,np.vstack([np.hstack(tiles[k:k+8]) for k in range(0,len(tiles),8)])); return meta
out={}
for lab,fn in (("urban","vehicles-2.mp4"),("highway","vehicles.mp4")):
    fr=env["sample_frames"]('data/'+fn,40); refs=json.load(open(f'outputs/cs2/refs_{fn}.json')); ext=[];mis=[]
    for f,rb in zip(fr,refs):
        rb=np.array(rb).reshape(-1,4); g=(rb,np.ones(len(rb))); b,c,k=pred(nano,f,640,.25); gh,ph,_=env["greedy_match"](g[0],b,c)
        for j in np.where(~ph)[0]: ext.append((f,b[j],c[j]))
        for i in np.where(~gh)[0]: mis.append((f,g[0][i],g[1][i]))
    print(lab,len(ext),len(mis))
    out[lab]={"extra":montage(ext,f'outputs/cs2/audit_{lab}_extra.jpg'),"missed":montage(mis,f'outputs/cs2/audit_{lab}_missed.jpg'),"n_extra":len(ext),"n_missed":len(mis)}
json.dump(out,open('outputs/cs2/audit_meta.json','w'))
