# Corners For Layout (CFL) PyTorch implementation

This is the adaptation of the Corners for Layout by C. Fernandez et.al (https://arxiv.org/abs/1903.08094)


This implementation has been tested under this environment:\
python 3.7 , 3.8 \
pytorch 1.4\
torchvision 0.5\
cuda 10.1

Run python3 train_CFL.py to train and python3 test_CFL.py to do inference with the EfficientNet based models.

pretrained weights to use with the TFCFL models:

[StdConvs](https://drive.google.com/file/d/1yiEV9PRzdaYpsDcd94yEWSI0rU3_fa9S/view?usp=sharing)

[EquiConvs](https://drive.google.com/file/d/1aPyFFyYUgbUugpG9Gnpr4DKUgmgR1jdh/view?usp=sharing)

Update 20-07-2020:
I have implemented a version of CFLPytorch called TFCFL which is CFLPytorch created by converting the model and the weights from TensorFlow CFL.

Run python3 test_TFCFL.py --conv_type Std --modelfile StdConvsTFCFL.pth to do inference with this model. 

Run python3 train_TFCFL.py to train with TFCFL

TensorFlow CFL metrics after fixing the threshold parameter (StdConvs):

EDGES: IoU: 0.564; Accuracy: 0.936; Precision: 0.696; Recall: 0.731; f1 score: 0.713

CORNERS: IoU: 0.553; Accuracy: 0.986; Precision: 0.687; Recall: 0.724; f1 score: 0.704

PyTorch TFCFL metrics (StdConvs): 

EDGES: IoU: 0.564; Accuracy: 0.936; Precision: 0.696; Recall: 0.731; f1 score: 0.713

CORNERS: IoU: 0.553; Accuracy: 0.986; Precision: 0.687; Recall: 0.724; f1 score: 0.704

PyTorch TFCFL metrics (EquiConvs):

EDGES: IoU: 0.536; Accuracy: 0.931; Precision: 0.679; Recall: 0.699; f1 score: 0.688

CORNERS: IoU: 0.539; Accuracy: 0.986; Precision: 0.690; Recall: 0.696; f1 score: 0.691



