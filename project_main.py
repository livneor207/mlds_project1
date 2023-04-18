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


# from torchmetrics.classification import MulticlassHammingDistance
# input2_argsort.requires_grad = True
# input1_argsort.requires_grad = False 

# metric = MulticlassHammingDistance(num_classes=5, task = '')

# loss = metric(input1_argsort , input2_argsort )
# loss.backward()


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
model_path = os.path.join(data_folder,  'model3.pth')
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
training_configuration.update_merics(loss_functions_name = 'ce', learning_rate = 1e-3,
                                     learning_type='self_supervised', batch_size= 100, 
                                     scheduler_name = 'ReduceLROnPlateau', max_opt = False,
                                     epochs_count = 10, perm= 'perm', num_workers = 0, 
                                     max_lr = 5e-3, hidden_size = 512, balance_factor = 0.25,
                                     amount_of_patch = 25)
device = training_configuration.device

# define data loaders 
"""
slice for debuging
"""



test_df = test_df[0:20]
train_df = train_df[0:25]
train_loader, val_loader, test_loader, debug_loader = \
    initialize_dataloaders(train_df, test_df, 
                           training_configuration, 
                           val_split=val_split,  
                           debug_batch_size = 8, 
                           random_state = seed, tb_writer = tb_writer)
# print size of data-sets
print(f'Train length = {train_loader.dataset.data_df.shape[0]}, val length = {val_loader.dataset.data_df.shape[0]}, test length = {test_loader.dataset.data_df.shape[0]}')


# set model 
model = CNN(training_configuration, 
             num_classes = amount_of_class,
             image_dim = (3,image_dim, image_dim))


student = generate_student(model, 
                           training_configuration, 
                           image_dim, 
                           amount_of_class)


summary(model, (3,image_dim, image_dim))
if not student is None:
    summary(student, (3,image_dim, image_dim))

# set optimizer 
optimizer, scheduler =  set_optimizer(model, training_configuration, train_loader, amount_of_class = amount_of_class, alpha = alpha)

# set accuracy metrics
accuracy_metric  = set_metric(training_configuration, amount_of_class = amount_of_class, metric_name = 'accuracy')
f_score_accuracy_metric  = set_metric(training_configuration, amount_of_class = amount_of_class, metric_name = 'f_score')

# set loss functions
if training_configuration.learning_type == 'supervised':
    criterion =  set_classifcation_loss(training_configuration, alpha = alpha)
else:    
    criterion=  set_similiarities_loss(classification_loss_name = 'CosineSimilarity')

ranking_criterion = set_rank_loss(loss_name = 'MarginRankingLoss', margin = 1, num_labels = 1)

# show example for data after transformations    
# generate data generation example
image, label, perm_order, class_name = generate_input_generation_examples(debug_loader)

train_results_df = main(model, student, optimizer, criterion,
                        ranking_criterion, accuracy_metric , 
                        train_loader, val_loader,
                        num_epochs=training_configuration.epochs_count,
                        device=device, 
                        tb_writer=tb_writer, 
                        max_opt = training_configuration.max_opt, 
                        model_path = model_path, 
                        scheduler = scheduler)


######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

training_configuration =  TrainingConfiguration()
training_configuration.get_device_type()

training_configuration.update_merics(loss_functions_name = 'ce', learning_rate = 1e-3,
                                     learning_type='supervised', batch_size= 100, 
                                     scheduler_name = 'ReduceLROnPlateau', max_opt = False,
                                     epochs_count = 10, perm= 'perm', num_workers = 0, 
                                     max_lr = 5e-3, hidden_size = 512, amount_of_patch = 25)


device = training_configuration.device

# define data loaders 
"""
slice for debuging
"""

train_loader, val_loader, test_loader, debug_loader = \
    initialize_dataloaders(train_df, test_df, 
                           training_configuration, 
                           val_split=val_split,  
                           debug_batch_size = 8, 
                           random_state = seed, tb_writer = tb_writer)
# print size of data-sets
# print size of data-sets
print(f'Train length = {train_loader.dataset.data_df.shape[0]}, val length = {val_loader.dataset.data_df.shape[0]}, test length = {test_loader.dataset.data_df.shape[0]}')



# model.load_state_dict(torch.load(model_path))
model_path = os.path.join(data_folder,  'model2.pth')

ssl_model =  SSLMODEL(model, 
                      num_classes=amount_of_class, 
                      image_dim=(3,image_dim, image_dim)
                      )
student= None

summary(ssl_model, (3,image_dim, image_dim))





# set optimizer 
optimizer, scheduler =  set_optimizer(ssl_model, training_configuration, train_loader, amount_of_class = amount_of_class, alpha = alpha)

# set accuracy metrics
accuracy_metric  = set_metric(training_configuration, amount_of_class = amount_of_class, metric_name = 'accuracy')
f_score_accuracy_metric  = set_metric(training_configuration, amount_of_class = amount_of_class, metric_name = 'f_score')

# set loss functions
if training_configuration.learning_type == 'supervised':
    criterion =  set_classifcation_loss(training_configuration, alpha = alpha)
else:    
    criterion=  set_similiarities_loss(classification_loss_name = 'CosineSimilarity')

ranking_criterion = set_rank_loss(loss_name = 'HingeEmbeddingLoss', margin = 1, num_labels = 1)

# show example for data after transformations    
# generate data generation example
image, label, perm_order, class_name = generate_input_generation_examples(debug_loader)


train_results_df = main(ssl_model, student, optimizer, criterion, ranking_criterion, accuracy_metric , 
                        train_loader, val_loader, num_epochs=training_configuration.epochs_count, device=device, 
                        tb_writer=tb_writer, max_opt = training_configuration.max_opt, model_path = model_path, scheduler = scheduler)








