from CFLPytorch.StdConvsCFL import StdConvsCFL
from CFLPytorch.EquiConvsCFL import EquiConvsCFL
from CFLPytorch.resnet import StdConvsCFL as Res50Std 
from CFLPytorch.StdConvsTFCFL import StdConvsTFCFL
from CFLPytorch.EquiConvsTFCFL import EquiConvsTFCFL
import argparse
import logging
#import sagemaker_containers
import os
import glob
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
from skimage.feature import corner_peaks, peak_local_max
import imageio
from offsetsTFCFL import offcalc

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


eps = 1e-10 # epsilon to improve numerical stability

def evaluate(map):
    
    if map == 'edges':
        prediction_path_list = glob.glob(os.path.join('test','EM_test')+'/*.jpg')
        gt_path_list = glob.glob(os.path.join('test', 'EM_gt')+'/*.jpg')
    if map == 'corners':
        prediction_path_list = glob.glob(os.path.join('test','CM_test')+'/*.jpg')
        gt_path_list = glob.glob(os.path.join('test', 'CM_gt')+'/*.jpg')
    prediction_path_list.sort()
    gt_path_list.sort()
    

    P, R, Acc, f1, IoU = [], [], [], [], []
    for im in range(len(prediction_path_list)):
        # predicted image
        prediction = Image.open(prediction_path_list[im])
        pred_W, pred_H = prediction.size
        prediction = np.array(prediction)/255.

        # gt image
        gt = Image.open(gt_path_list[im])
        gt = gt.resize([pred_W, pred_H])
        gt = np.array(gt)/255.
        gt = (gt>=0.1).astype(int)

        th=0.1
        tp = np.sum(np.logical_and(gt==1,prediction>th))
        tn = np.sum(np.logical_and(gt==0,prediction<=th))
        fp = np.sum(np.logical_and(gt==0,prediction>th))
        fn = np.sum(np.logical_and(gt==1,prediction<=th))

        # How accurate the positive predictions are
        P.append(tp / (tp + fp))
        # Coverage of actual positive sample
        R.append(tp / (tp + fn))
        # Overall performance of model
        Acc.append((tp + tn) / (tp + tn + fp + fn))
        # Hybrid metric useful for unbalanced classes 
        f1.append(2 * (tp / (tp + fp))*(tp / (tp + fn))/((tp / (tp + fp))+(tp / (tp + fn))))
        # Intersection over Union
        IoU.append(tp / (tp + fp + fn))

    return np.mean(P), np.mean(R), np.mean(Acc), np.mean(f1), np.mean(IoU)

class SUN360Dataset(Dataset):
    

    def __init__(self, file, transform=None, target_transform=None, joint_transform=None):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            transform (callable, optional): Optional transform to be applied
                on an image.
            target_file (callable, optional): Optional transform to be applied
                on a map (edge and corner).    
        """
    
        self.images_data = pd.read_json(file)    
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images_data.iloc[idx, 0]                        
        EM_name = self.images_data.iloc[idx, 1]
        CM_name = self.images_data.iloc[idx, 2]
        CL_name = self.images_data.iloc[idx, 3]
        H = 128
        W = 256
        C = 3
        predH = 64
        predW = 128
        
        image = Image.open(img_name)
        if image.mode !='RGB' :
            image = image.convert('RGB')
        image = image.resize([W, H], Image.ANTIALIAS)
        image = np.array(image).astype('float32')
        #image = np.expand_dims(np.asarray(image), axis = 0)
        mean_color = [103.939, 116.779, 123.68]
        r,g,b = np.split(image,3,axis=2)
        bgr = np.concatenate((b - mean_color[0], g - mean_color[1], r - mean_color[2]),axis=2)
        image = torch.from_numpy(bgr)
        image = image.permute(2,0,1)
        

        EM = Image.open(EM_name)
        EM = EM.resize([predW, predH])
        EM = np.array(EM).astype('float32')/255.
        EM = np.expand_dims(np.asarray(EM), axis = 0)
        EM = torch.from_numpy(EM)
        #EM = EM.permute(0,3,1,2)
        
        CM = Image.open(CM_name)
        CM = CM.resize([predW, predH])
        CM = np.array(CM).astype('float32')/255.
        CM = np.expand_dims(np.asarray(CM), axis = 0)
        CM = torch.from_numpy(CM)
        #CM = CM.permute(0,3,1,2)
        
        with open(CL_name, mode='r') as f:
            cor = np.array([line.strip().split() for line in f], np.int32)
        if(len(cor)%2 != 0) :
            print (CL_name.split('/')[-1])    
        
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            CM = self.target_transform(CM)
            EM = self.target_transform(EM)
        if self.joint_transform is not None:   
            image, EM, CM, cor = self.joint_transform([image, EM, CM, cor])      
        
        return image, EM, CM, img_name


def corners_2_xy(outputs):
    output = outputs['output_likelihood'] 
    output = torch.sigmoid(output)
    edges,corners =torch.chunk(output,2,dim=1)
    corner1= 255* corners
    corner1[corner1>127] = 255
    corner1[corner1<127] = 0
    corner1 = torch.cat((corner1,corner1,corner1),dim=-1)
    corner1 = torch.squeeze(corner1)
    array = corner1.detach().cpu().numpy().astype(np.uint8)
    local_peaks = corner_peaks(array, min_distance=5, threshold_rel=0.5, indices=True)
    local_peaks = np.array(local_peaks, dtype=np.float64)
    height, width = array.shape
    width /=3
    col1m = (local_peaks[:,1]>=width) & (local_peaks[:,1]<2*width)
    peaks = local_peaks[col1m] 
    peaks[:,0]/=height
    peaks[:,1]-= width
    peaks[:,1]/= width
    return peaks 


def map_predict(outputs, EM_gt,CM_gt):
    '''
    function to calculate total loss according to CFL paper
    '''
    output= outputs['output_likelihood']
    edges,corners =torch.chunk(output,2,dim=1)
    edges, corners = torch.sigmoid(edges), torch.sigmoid(corners)
    EM = EM_gt
    CM = CM_gt
    
    #edges,corners = torch.squeeze(edges,dim=1), torch.squeeze(corners,dim=1) 
    #EM,CM = torch.squeeze(EM,dim=1), torch.squeeze(CM,dim=1)
    P_e, R_e, Acc_e, f1_e, IoU_e = evaluate(edges,EM)
    #print('EDGES: IoU: ' + str('%.3f' % IoU_e) + '; Accuracy: ' + str('%.3f' % Acc_e) + '; Precision: ' + str('%.3f' % P_e) + '; Recall: ' + str('%.3f' % R_e) + '; f1 score: ' + str('%.3f' % f1_e))
    P_c, R_c, Acc_c, f1_c, IoU_c = evaluate(corners, CM)
    #print('CORNERS: IoU: ' + str('%.3f' % IoU_c) + '; Accuracy: ' + str('%.3f' % Acc_c) + '; Precision: ' + str('%.3f' % P_c) + '; Recall: ' + str('%.3f' % R_c) + '; f1 score: ' + str('%.3f' % f1_c))
    
    P_e_np = P_e.cpu().numpy()
    R_e_np = R_e.cpu().numpy()
    Acc_e_np = Acc_e.cpu().numpy()
    f1_e_np = f1_e.cpu().numpy()
    IoU_e_np = IoU_e.cpu().numpy()
    
    P_c_np = P_c.cpu().numpy()
    R_c_np = R_c.cpu().numpy()
    Acc_c_np = Acc_c.cpu().numpy()
    f1_c_np = f1_c.cpu().numpy()
    IoU_c_np = IoU_c.cpu().numpy() 

    return P_e_np, R_e_np, Acc_e_np, f1_e_np, IoU_e_np, P_c_np, R_c_np, Acc_c_np, f1_c_np, IoU_c_np

def _test(args):
    """
    is_distributed = len(args.hosts) > 1 and args.dist_backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.dist_backend, rank=host_rank, world_size=world_size)
        logger.info(
            'Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
                args.dist_backend,
                dist.get_world_size()) + 'Current host rank is {}. Using cuda: {}. Number of gpus: {}'.format(
                dist.get_rank(), torch.cuda.is_available(), args.num_gpus))
    """            

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    
    logger.info("Device Type: {}".format(device))
    img_size= [128,256]
    logger.info("Loading SUN360 dataset")
    """
    transform = transforms.Compose(
        [transforms.Resize((img_size[0],img_size[1])),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.458, 0.408], std=[1.0, 1.0, 1.0])])
    target_transform = transforms.Compose([transforms.Resize((img_size[0],img_size[1])),
                                           transforms.ToTensor()])     
    """                                       

    testset = SUN360Dataset("testdata.json",transform = None, target_transform = None)
    test_loader = DataLoader(testset, batch_size=args.batch_size,
                                               shuffle=False, num_workers=args.workers)
                                              
    
    logger.info("Model loaded")
    if args.conv_type == "Std":
        #model = StdConvsCFL(args.model_name,conv_type=args.conv_type, layerdict=None, offsetdict=None)
        #model = Res50Std()
        model = StdConvsTFCFL()
    elif args.conv_type == "Equi":                           
        layerdict, offsetdict = offcalc(args.batch_size)
        #model = EquiConvsCFL(args.model_name,conv_type=args.conv_type, layerdict=layerdict, offsetdict=offsetdict)
        model = EquiConvsTFCFL(layerdict=layerdict, offsetdict=offsetdict)
    #model.load_state_dict(torch.load(args.modelfile))
    pretrained_dict = torch.load(args.modelfile)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    if not os.path.exists(os.path.join('test','EM_test')): os.makedirs(os.path.join('test','EM_test'))
    if not os.path.exists(os.path.join('test','CM_test')): os.makedirs(os.path.join('test','CM_test'))
    with torch.no_grad():
        model = model.to(device)
        
        for i, data in enumerate(test_loader):
            # get the inputs
            inputs, EM , CM, image_name = data
            inputs = inputs.to(device)
            model.eval()
            outputs = model(inputs)
            output = outputs['output_likelihood']
            edges, corners = torch.chunk(output, 2 , dim=1)
            edges = torch.sigmoid(edges)
            emap = edges.permute(0,2,3,1).cpu().numpy()
            corners = torch.sigmoid(corners)
            cmap = corners.permute(0,2,3,1).cpu().numpy()
            filename = image_name[0]
            filename = filename.split('.')[0]
            filename = filename.split('/')[-1]
            imageio.imwrite(os.path.join('test','EM_test',filename + "_emap.jpg"), emap[0,:,:,0])
            imageio.imwrite(os.path.join('test','CM_test',filename + "_cmap.jpg"), cmap[0,:,:,0])

            #detection= corners_2_xy(outputs)
            #print(len(detection))
             
        
    P_e, R_e, Acc_e, f1_e, IoU_e = evaluate('edges')
    print('EDGES: IoU: ' + str('%.3f' % IoU_e) + '; Accuracy: ' + str('%.3f' % Acc_e) + '; Precision: ' + str('%.3f' % P_e) + '; Recall: ' + str('%.3f' % R_e) + '; f1 score: ' + str('%.3f' % f1_e))
    P_c, R_c, Acc_c, f1_c, IoU_c = evaluate('corners')
    print('CORNERS: IoU: ' + str('%.3f' % IoU_c) + '; Accuracy: ' + str('%.3f' % Acc_c) + '; Precision: ' + str('%.3f' % P_c) + '; Recall: ' + str('%.3f' % R_c) + '; f1 score: ' + str('%.3f' % f1_c))
    print('Finished Testing')
              


def _save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


def model_fn(model_dir):
    logger.info('model_fn')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StdConvsCFL('efficient-b0',conv_type='Std')
    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--batch_size', type=int, default=1, metavar='BS',
                        help='batch size (default: 1)')
    parser.add_argument('--model-dir', type=str, default="")
    parser.add_argument('--model-name', type=str, default="efficientnet-b0")
    parser.add_argument('--conv_type', type=str,default="Std", help='select convolution type between Std and Equi. Also determines the network type')
    parser.add_argument('--modelfile', type=str, default="model_epoch100.pth", help="load model file for inference")
    #parser.add_argument('--dist_backend', type=str, default='gloo', help='distributed backend (default: gloo)')

    #env = sagemaker_containers.training_env()
    #parser.add_argument('--hosts', type=list, default=env.hosts)
    #parser.add_argument('--current-host', type=str, default=env.current_host)
    #parser.add_argument('--model-dir', type=str, default=env.model_dir)
    #parser.add_argument('--data-dir', type=str, default=env.channel_input_dirs.get('training'))
    #parser.add_argument('--num-gpus', type=int, default=env.num_gpus)

    _test(parser.parse_args())