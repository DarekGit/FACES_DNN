import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt

def IoU(ground_box,det_box):
  l,t,r,b = ground_box[0:4]
  ll,tt,rr,bb = det_box[0:4]
  lstar = max(l,ll)
  rstar = min(r,rr)
  if lstar>rstar:
    return 0. # since there is no intersection
  tstar = max(t,tt)
  bstar = min(b,bb)
  if tstar>bstar:
    return 0. # since there is no intersection
  
  garea = (r-l+1)*(b-t+1)
  darea =(rr-ll+1)*(bb-tt+1)
  iarea = (rstar-lstar+1)*(bstar-tstar+1)
  return iarea/(garea+darea-iarea)

# Uzupelnianie metryki wynikow, wyliczenie IoU, wybor TruePositive oraz FP i FN
def Classifcation(gts,dts):
  xg=len(gts)
  yd=len(dts)
  if xg*yd >0:
      MIoU =np.zeros((yd,xg))/1.
      for i in range(yd*xg): #wyliczenie IoU dla macierzy y detekcji i x anotacji (real)
        MIoU[i//xg,i%xg]=IoU(gts[i%xg],dts[i//xg])
      TP=[]
      FP=list(np.arange(yd)/1.)
      TN=list(np.arange(xg)/1.) 
      for i in range(xg): # wyszukanie maksymalnych IoU >0 dla x rzeczywistych twarzy (annotation)
        a=np.argmax(MIoU)
        k,w =a//xg,a%xg
        if MIoU[k,w]>0:
          TP.append([k/1.,w/1.,MIoU[k,w]])
          MIoU[:,w]=0.
          MIoU[k,:]=0.
          FP[k]=-1.
          TN[w]=-1.
      FP=[e for e in FP if e>-1]
      TN=[e for e in TN if e>-1]
  else: #no intersections
    TP=[]
    if xg>0:
      TN=list(np.arange(xg)/1.)
      FP=[] #brak detekcji
    else:
      FP=list(np.arange(yd)/1.)
      TN=[]  #brak anotacji
  return TP,TN,FP

#flat list of ground true and detected boxes sorted by decreasing confidence 
def lists(gbxs,dbxs,conf_t=0):
  dts=[[[(bx[2]-bx[0]+1)*(bx[3]-bx[1]+1),*bx,0,-1] for bx in img if bx[4]>=conf_t ] for img in dbxs]
  gts=[[[(bx[2]-bx[0]+1)*(bx[3]-bx[1]+1),*bx] for bx in img] for img in gbxs]
  dts_f=[]
  gts_f=[]
  for i,img in enumerate(dts):
    TP,_,_=Classifcation([[*np.array(bx)[1:5]] for bx in gts[i]],[[*np.array(bx)[1:5]] for bx in img])
    for k,w,IoU in TP:
      k=int(k)
      w =int(w)
      img[k][6]=IoU #IoU
      img[k][7]=w #wskaznik box for gbx
      img[k][0]=gts[i][w][0] #size for box from gbx
    for bx in img:
      dts_f.append(bx)
  dts_f=sorted(dts_f,key=itemgetter(5),reverse=True)
  for img in gts:
    for bx in img:
      gts_f.append(bx)
  return gts_f,dts_f


def AP_R(dts_f,gt_nb,IoU_t=0.5,data=False):
  r_p_full=[]
  det_nb=0
  tp_nb=0
  for bx in dts_f: #narastajace wyliczenie precission TP/Det dla recall wzgledem wszystkich GT 
    det_nb+=1
    if bx[6]>IoU_t:
      tp_nb+=1
    r_p_full.append([tp_nb/gt_nb,tp_nb/det_nb])
  r_p_full=sorted(r_p_full,key=itemgetter(0))
  rM=-1; r_p=[]
  for r,p in r_p_full: #uszeregowanie po recall
    if r>rM:
      r_p.append([r,p])
      rM=r
    else: 
      if p>r_p[-1][1]:
        r_p[-1][1]=p
  r_p_rev=np.array(r_p[::-1])
  i=-1
  r_p_int=[]
  while i>0 or i==-1: #lista recall dla kolejnych maksymów precission
    if len(r_p_rev[0:i]) >0:
      i=np.argmax(r_p_rev[0:i,1])
      r_p_int.append([*r_p_rev[i]])
    else: 
      r_p_int.append([0,0])
      break

  AP=0
  rp=0
  for r,p in r_p_int:
    AP+=(r-rp)*p
    rp=r
  if data:
    return (AP,r_p_int[-1][0],IoU_t),(r_p_int,r_p_full)
  else:
    return (AP,r_p_int[-1][0],IoU_t),()


def mAP(gbxs,dbxs,conf_t=0,IoUs=[],data=False):
  mAP={}
  Data={}
  if IoUs is None:
    IoUs=[.5]
  if IoUs==[] or type(IoUs)!=list:
    IoUs=[.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.0]
  gts,dts=lists(gbxs,dbxs,conf_t=conf_t)
  for IoU in IoUs:
    key='All ' + '{:.2f}'.format(IoU)
    mAP[key],Data[key]=AP_R(dts,len(gts),IoU_t=IoU,data=data)
  
  gt=[x for x in gts if x[0]<=32**2]
  dt=[x for x in dts if x[0]<=32**2]
  mAP['small'],Data['small']=AP_R(dt,len(gt),IoU_t=IoUs[0],data=data)

  gt=[x for x in gts if x[0]>32**2 and x[0]<=96**2]
  dt=[x for x in dts if x[0]>32**2 and x[0]<=96**2]
  mAP['medium'],Data['medium']=AP_R(dt,len(gt),IoU_t=IoUs[0],data=data)

  gt=[x for x in gts if x[0]>96**2]
  dt=[x for x in dts if x[0]>96**2]
  mAP['large'],Data['large']=AP_R(dt,len(gt),IoU_t=IoUs[0],data=data) 

  return mAP,Data


def plot_mAP(met,data,keys,r_p=1,title='',file='mAP',figsize=(16,10)):
  fig, ax = plt.subplots(figsize=figsize)
  legend=[]
  if r_p!=1:
    r_p=0
  if keys=='All':
    keys=met.keys()
  for k in keys:
    if k in data.keys():
      print('{:8}:   AP: {:5.2f}%   Recall: {:5.2f}%   IoU: {:.2f}'.format(k,met[k][0]*100,met[k][1]*100,met[k][2]))
      legend.append('{:8}:   {:5.2f}% / {:5.2f}%   -  {:.2f}'.format(k,met[k][0]*100,met[k][1]*100,met[k][2]))
      r=np.array(data[k][r_p])[:,0]
      p=np.array(data[k][r_p])[:,1]
      ax.plot(r,p )
  title+=' mAP'
  if r_p==1:
    title+=' - no estimation'
  plt.legend(legend, loc='lower left')
  ax.set(xlabel='Recall', ylabel='Precission',title=title)
  ax.grid()
  file=file.split('.')[0]+'.png'
  fig.savefig(file)
  plt.show()

#lista po obrazach list boxow [[[l,t,b,r]*n]*imgs], dla detected z confidence [[[l,t,b,r,conf]*n]*imgs]


