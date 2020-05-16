import pandas as pd
import os

imgs = os.listdir('trainsmall/RGB')
CMimgs = os.listdir("trainsmall/CM_gt")
EMimgs = os.listdir("trainsmall/EM_gt")


imgpaths = []
CMpaths = []
EMpaths = []
root=""
#abspth = os.path.abspath(root)
for img in imgs:
    pths = os.path.join(root,"trainsmall/RGB/",img)
    imgpaths.append(pths)
for CMimg in CMimgs:
    pths = os.path.join(root,"trainsmall/CM_gt/",CMimg)
    CMpaths.append(pths)
for EMimg in EMimgs:
    pths = os.path.join(root,"trainsmall/EM_gt/",EMimg)
    EMpaths.append(pths)    
dict={'images' : imgpaths, 'EM' : EMpaths, 'CM' : CMpaths}
df = pd.DataFrame(data = dict)
df.to_json("traindatasmall.json")

cornerimages = os.listdir("train/morethan4corners")
corners=[]
for item in cornerimages:
    item=item.split('_')
    name= item[0]+'_'+item[1]
    corners.append(name)

imgpaths = []
CMpaths = []
EMpaths = []
root=""

for img in corners:
    pths = os.path.join(root,"train/RGB/",img+".jpg")
    imgpaths.append(pths)
    pths = os.path.join(root,"train/CM_gt/",img+"_CM.jpg")
    CMpaths.append(pths)
    pths = os.path.join(root,"train/EM_gt/",img+"_EM.jpg")
    EMpaths.append(pths)
    
  
dict={'images' : imgpaths, 'EM' : EMpaths, 'CM' : CMpaths}
df = pd.DataFrame(data = dict)
df.to_json("morethan4corners.json")