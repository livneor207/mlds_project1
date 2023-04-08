import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import copy
import pandas as pd
import torch.nn.utils.prune as prune
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import torchmetrics
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np 
from albumentations import Compose, Normalize, RandomCrop, HorizontalFlip, ShiftScaleRotate, HueSaturationValue
import cv2
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch.nn as nn
from torchsummary import summary
import pandas as pd
import numpy as np 
from torch.utils.data import Subset
import random
import seaborn as sns 
import warnings
import gc
import os
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import collections
import time
import copy
import sys
import os
from sklearn.metrics import f1_score
from torchmetrics import F1Score
from torch.nn import functional as F
from focal_loss.focal_loss import FocalLoss
from torch.utils.tensorboard import SummaryWriter
from tensorboard_hellper import *
from utils import *
from data_set_loader import *
from train_hellper import *
from model_builder import *
import glob
plt.close('all')





"""
dash board tensorboard 
tensorboard --logdir logdir_folder_path --port default
tensorboard --logdir "C:\MSC\opencv-python-free-course-code\classification_project\opencv-pytorch-dl-course-classification\expirement1" --port default
"""

# seed
seed =  48
val_split = 0.1
image_dim = 224

seed_everything(seed)

# set path's
current_folder = os.getcwd()
data_folder = os.path.join(current_folder,  'dogs-vs-cats')
test_folder_path = os.path.join(data_folder,  'test1', 'test1')
train_folder_path = os.path.join(data_folder,  'train', 'train')
log_dir = os.path.join(data_folder,  'expirement1')
submission_path = os.path.join(data_folder,  'submission.csv')

# get all files names 
train_images_path_list = get_all_images_from_specific_folder(train_folder_path)
test_images_path_list = get_all_images_from_specific_folder(test_folder_path)

# parse train data
train_df = parse_train_data(train_images_path_list)
train_df = train_df.sample(frac=1).reset_index(drop=True)

test_df = parse_train_data(train_images_path_list)

# log to tensorboard, tensorboard summary writer
tb_writer = SummaryWriter(
    log_dir = log_dir,
    comment = "TensorBoard in PyTorch")

# get data statistics
train_statistic_df, alpha = data_statistics(train_df)

# get statistic of data
class_ratios, amount_of_class = get_statistic_from_stistic_dataframe(train_statistic_df)

# generate histogram to illustate if data is balance or not 
generate_hitogram_base_dataframe_column(train_df, 'class_name')

# set train configurations
training_configuration =  TrainingConfiguration()
training_configuration.get_device_type()
training_configuration.update_merics(loss_functions_name = 'FL', learning_rate = 1e-3)
device = training_configuration.device

# define data loaders 
"""
slice for debuging
"""
# test_df = test_df[0:20]
# train_df = train_df[0:100]
train_loader, val_loader, test_loader, debug_loader = \
    initialize_dataloaders(train_df, test_df,  \
                           batch_size = training_configuration.batch_size, val_split=val_split,  \
                               debug_batch_size = 8, random_state = seed, tb_writer = tb_writer, taske_name = 'no_perm')
# print size of data-sets
print(f'Train length = {train_loader.dataset.data_df.shape[0]}, val length = {val_loader.dataset.data_df.shape[0]}, test length = {test_loader.dataset.data_df.shape[0]}')



# set model 
model =  CNN(num_classes = amount_of_class, image_dim = (3,image_dim, image_dim))    



# set optimizer 
optimizer, scheduler =  set_optimizer(model, training_configuration, train_loader, amount_of_class = amount_of_class, alpha = alpha)

# set accuracy metrics
accuracy_metric  = set_metric(training_configuration, amount_of_class = amount_of_class, metric_name = 'accuracy')
f_score_accuracy_metric  = set_metric(training_configuration, amount_of_class = amount_of_class, metric_name = 'f_score')

# set loss functions
classification_criterion =  set_classifcation_loss(training_configuration, alpha = alpha)




EPOCHS = 50
BATCHES = 100
steps = []
lrs = []
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9) # Wrapped optimizer
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=BATCHES, epochs=EPOCHS)

for epoch in range(EPOCHS):
    for batch in range(BATCHES):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
        steps.append(epoch * BATCHES + batch)

plt.figure()
plt.legend()
plt.plot(steps, lrs, label='OneCycle')
plt.show()


# show example for data after transformations    
# generate data generation example
image, label, perm_order, class_name = generate_input_generation_examples(debug_loader)

train_results_df = main(model, optimizer, classification_criterion, accuracy_metric , 
                        train_loader, val_loader, num_epochs=1, device=device, 
                        tb_writer=tb_writer)












