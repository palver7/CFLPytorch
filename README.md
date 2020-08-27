# Corners For Layout (CFL) PyTorch implementation

This is the adaptation of the Corners for Layout by C. Fernandez et.al (https://arxiv.org/abs/1903.08094)


This implementation has been tested under this environment:\
python 3.8\
pytorch 1.4\
torchvision 0.5\
cuda 10.1

run python3 train_CFL.py to train and test_CFL.py to do inference with the EfficientNet based models.

Update 20-07-2020:
I have implemented a version of CFLPytorch called TFCFL which is CFLPytorch created by converting the model and the weights from TensorFlow CFL.
run python3 test_TFCFL.py to do inference with this model. 


