import pandas as pd
import os
"""
imgs = os.listdir('test/RGB')
CMimgs = os.listdir("test/CM_gt")
EMimgs = os.listdir("test/EM_gt")
corlabs = os.listdir("test/corner_labels")


imgpaths = []
CMpaths = []
EMpaths = []
corlabpaths=[]
root=""
#abspth = os.path.abspath(root)
for img in imgs:
    pths = os.path.join(root,"test/RGB/",img)
    imgpaths.append(pths)
for CMimg in CMimgs:
    pths = os.path.join(root,"test/CM_gt/",CMimg)
    CMpaths.append(pths)
for EMimg in EMimgs:
    pths = os.path.join(root,"test/EM_gt/",EMimg)
    EMpaths.append(pths)
for corlab in corlabs:
    pths = os.path.join(root,"test/corner_labels/",corlab)
    corlabpaths.append(pths)       
  
dict={'images' : imgpaths, 'EM' : EMpaths, 'CM' : CMpaths, 'CL' : corlabpaths}
df = pd.DataFrame(data = dict)
df.to_json("testdata.json")
"""

cornerimages = os.listdir("train/morethan4corners")
corners=[]
for item in cornerimages:
    item=item.split('_')
    name= item[0]+'_'+item[1]
    corners.append(name)

imgpaths = []
CMpaths = []
EMpaths = []
corlabpaths=[]
root=""

for img in corners:
    pths = os.path.join(root,"train/RGB/",img+".jpg")
    imgpaths.append(pths)
    pths = os.path.join(root,"train/CM_gt/",img+"_CM.jpg")
    CMpaths.append(pths)
    pths = os.path.join(root,"train/EM_gt/",img+"_EM.jpg")
    EMpaths.append(pths)
    pths = os.path.join(root,"train/corner_labels/",img+".txt")
    corlabpaths.append(pths)
    
  
dict={'images' : imgpaths, 'EM' : EMpaths, 'CM' : CMpaths, 'CL' : corlabpaths}
df = pd.DataFrame(data = dict)
df.to_json("morethan4corners.json")
