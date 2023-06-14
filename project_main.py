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
import itertools
import os
from sklearn.metrics import f1_score
from torchmetrics import F1Score
from torch.nn import functional as F
from focal_loss.focal_loss import FocalLoss
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tensorboard_hellper import *
from utils import *
from data_set_loader import *
from train_hellper import *
from model_builder import *
import glob
plt.close('all')
    

"""
# TODO! remove features.7.4.block.3
# TODO! add to training configuration in colab
# TODO! unuse\run perm head

"""




"""
dash board tensorboard 
tensorboard --logdir logdir_folder_path --port default
tensorboard --logdir "C:\MSC\opencv-python-free-course-code\classification_project\opencv-pytorch-dl-course-classification\expirement1" --port default
"""

# seed
seed =  48
val_split = 0.001
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


task_name  = 'OxfordIIITPet'

task_name  = 'cat_dogs'
task_name  = 'CIFAR10'
task_name  = 'OxfordIIITPet'
task_name  = 'cat_dogs'
task_name  = 'cat_dogs'
task_name  = 'CIFAR10'
task_name  = 'OxfordIIITPet'

if task_name in ['CIFAR10', 'cat_dogs']:
    train_df, train_data= parse_train_data(task_name  =task_name, folder_path =train_folder_path, train=True, current_folder= current_folder)
    test_df, test_data = parse_train_data(task_name=task_name, folder_path =test_folder_path, train=False, current_folder = current_folder)
elif task_name == 'OxfordIIITPet':
    train_df, train_data= parse_train_data(task_name  =task_name, folder_path =train_folder_path, train='trainval', current_folder= current_folder)
    test_df, test_data = parse_train_data(task_name=task_name, folder_path =test_folder_path, train='test', current_folder = current_folder)
# parse train data

# 'trainval'
# 'test'


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
training_configuration.update_merics(loss_functions_name = 'ce', learning_rate = 1e-4,
                                     learning_type='self_supervised', batch_size= 20, 
                                     scheduler_name = 'None', max_opt = False,
                                     epochs_count = 100, perm= 'perm', num_workers = 0, 
                                     max_lr = 5e-3, hidden_size = 512, balance_factor = 1,
                                     balance_factor2 = 1, amount_of_patch = 4, 
                                     moving_average_decay = 0.995,weight_decay = 1e-6, 
                                     optimizer_name = 'lion', max_allowed_permutation = 100)


device = training_configuration.device

# define data loaders 
"""
slice for debuging
"""
# amount_for_debug = 200
# test_df = test_df[0:amount_for_debug]
# train_df = train_df[0:amount_for_debug]
# if train_data is not None:
#     train_data = train_data[0:amount_for_debug]
#     test_data = test_data[0:amount_for_debug]

train_loader, val_loader, test_loader, debug_loader = \
    initialize_dataloaders(train_df, test_df, 
                           training_configuration, 
                           val_split=val_split,  
                           debug_batch_size = 8, 
                           random_state = seed,
                           tb_writer = tb_writer,
                           train_data=train_data,
                           test_data=test_data,
                           image_size = image_dim,
                           rand_choise = False,
                           orig_pe = True,
                           train_split = 0.05)
    
# print size of data-sets
print(f'Train length = {train_loader.dataset.data_df.shape[0]}, \
      val length = {val_loader.dataset.data_df.shape[0]},  \
      test length = {test_loader.dataset.data_df.shape[0]}')



model = CNN(training_configuration, 
              num_classes = amount_of_class,
              image_dim = (3,image_dim, image_dim),
              freeze_all=False, 
              model_name = 'resnet50',
              weights='IMAGENET1K_V1',
              unfreeze=False)




# from sklearn.manifold import TSNE

# image, label, perm_order, class_name = generate_input_generation_examples(train_loader, debug = False)
# image1 = image[:,0:3,:,:]
# image2 = image[:,3::,:,:]

# embbeding1, perm_order_pred1  = model(image1)
# embbeding2, perm_order_pred2  = model(image2)


# perms_t = perm_order[:,0,:]
# y_label = label.argmax(1)
# y_label= y_label.detach().numpy()
# # set model 
# TSNE_model = TSNE(n_components=2, learning_rate='auto',
#                     init='random', perplexity=10, n_iter= 1000)

# # train model
# projection = TSNE_model.fit_transform(embbeding.detach().numpy())




# plt.figure()
# plt.title('TNSE-2D')
# plt.scatter(projection[:, 0], projection[:, 1], c=y_label, s=30, cmap='Set1')
# plt.grid()
# plt.show()


student = generate_student(model, 
                           training_configuration, 
                           image_dim, 
                           amount_of_class,
                           model_name = 'resnet50',
                           weights = None,
                           unfreeze = True)

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
    criterion=  set_similiarities_loss(classification_loss_name = 'CosineSimilarity', beta = 1)

ranking_criterion = set_rank_loss(loss_name = 'CosineSimilarity', margin = 1, num_labels = 1, beta = 1)
perm_creterion = nn.CrossEntropyLoss()

# show example for data after transformations    
# generate data generation example
image, label, perm_order, class_name, perm_label = generate_input_generation_examples(debug_loader)

train_results_df = main(model, student, optimizer, criterion,
                        ranking_criterion, accuracy_metric , perm_creterion,
                        train_loader, val_loader,
                        num_epochs=training_configuration.epochs_count,
                        device=device, 
                        tb_writer=tb_writer, 
                        max_opt = training_configuration.max_opt, 
                        model_path = model_path, 
                        scheduler = scheduler)
aa =4 
# import torch
# import torch.nn as nn

# y_true = torch.Tensor([[1, 2, 3],[3,2,1]]) # true tensor
# y_pred = torch.Tensor([[0.2, 0.5, 0.1],[-0.1,0.2,0.3]]) # predicted tensor
# y_pred.requires_grad = True
# # Apply softmax activation to both tensors
# temperature = 30
# temperature, dummy = y_pred.max(1)
# temperature2, dummy = y_true.max(1)


# # Apply softmax activation to both tensors
# true_probs = nn.functional.softmax(y_true, dim=0)



# pred_probs_ordered = nn.functional.softmax(torch.div(y_pred.T, temperature).T, dim=0)
# y_true_probs_ordered = nn.functional.softmax(torch.div(y_true.T, temperature2).T, dim=0)

# # Compute the KL divergence between the true and predicted probabilities
# criterion = nn.KLDivLoss(reduction='batchmean')
# loss = criterion(torch.log(pred_probs_ordered), y_true_probs_ordered)

# print(loss) # this should output the same loss value as for [3,2,1]



# output = torch.Tensor([300,200,100]) # predicted output
# target = torch.Tensor([1.0,2.0,3.0]) # true label

# criterion = nn.SoftMarginLoss()
# criterion = nn.HingeEmbeddingLoss()
# criterion = nn.MultiLabelSoftMarginLoss()
# criterion = nn.MultiLabelMarginLoss()

# loss = criterion(output, target) # compute the loss



# a=5 
# import torch.nn.functional as F
# kl_loss = nn.KLDivLoss(reduction="batchmean")
# target = torch.tensor([[2,1,3,4,5], [2,1,3,4,5], [2,1,3,4,5]],requires_grad = True, dtype = torch.float)
# pred =  torch.tensor([[2,1,5,3,4], [2,4,3,4,1], [2,1,3,4,5]],requires_grad = True, dtype = torch.float)
# pred = F.log_softmax(pred, dim=1)
# target = F.softmax(target, dim=1)
# output = kl_loss(pred, target)


######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
model_path = os.path.join(data_folder,  'model3.pth')

training_configuration =  TrainingConfiguration()
training_configuration.get_device_type()




training_configuration.update_merics(loss_functions_name = 'ce', learning_rate = 1e-4,
                                      learning_type='self_supervised', batch_size= 16,  
                                      scheduler_name = 'None', max_opt = True,
                                      epochs_count = 50, perm= 'perm', num_workers = 0, 
                                      max_lr = 5e-3, hidden_size = 512, balance_factor = 1,
                                      amount_of_patch = 4, moving_average_decay = 0.996,
                                      weight_decay=0, optimizer_name = 'lion')


device = training_configuration.device

model = CNN(training_configuration, 
              num_classes = amount_of_class,
              image_dim = (3,image_dim, image_dim),
              freeze_all=False, 
              model_name = 'resnet50',
              weights=None,
              unfreeze=False)


training_configuration.learning_type = 'supervised'

# define data loaders 
"""
slice for debuging
"""

train_loader, val_loader, test_loader, debug_loader = \
    initialize_dataloaders(train_df, test_df, 
                            training_configuration, 
                            val_split=val_split,  
                            debug_batch_size = 8, 
                            random_state = seed,
                            tb_writer = tb_writer,
                            train_data=train_data,
                            test_data=test_data,
                            image_size = image_dim,
                            rand_choise = False,
                            orig_pe = True,
                            train_split = 0.05)
# print size of data-sets
# print size of data-sets
print(f'Train length = {train_loader.dataset.data_df.shape[0]}, val length = {val_loader.dataset.data_df.shape[0]}, test length = {test_loader.dataset.data_df.shape[0]}')


model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# model.load_state_dict(torch.load(model_path))
model_path = os.path.join(data_folder,  'model2.pth')

ssl_model =  SSLMODEL(model, 
                      num_classes=amount_of_class, 
                      image_dim=(3,image_dim, image_dim),
                      freeze_all = False,
                      model_name = 'resnet50')
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
    criterion = torch.nn.MSELoss()
ranking_criterion = set_rank_loss(loss_name = 'CosineSimilarity', margin = 1, num_labels = 1)
perm_creterion = nn.CrossEntropyLoss()

# show example for data after transformations    
# generate data generation example
image, label, perm_order, class_name, perm_label = generate_input_generation_examples(debug_loader)


train_results_df = main(ssl_model, student, optimizer, criterion, ranking_criterion, accuracy_metric , perm_creterion,
                        train_loader, val_loader, num_epochs=training_configuration.epochs_count, device=device, 
                        tb_writer=tb_writer, max_opt = training_configuration.max_opt, model_path = model_path, scheduler = scheduler)



# ######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ######### zero shot learning ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# """
# fuuly - supervised
# """


training_configuration =  TrainingConfiguration()
training_configuration.get_device_type()




training_configuration.update_merics(loss_functions_name = 'ce', learning_rate = 1e-3,
                                      learning_type='supervised', batch_size= 16, 
                                      scheduler_name = 'None', max_opt = True,
                                      epochs_count = 50, perm= 'perm', num_workers = 0, 
                                      max_lr = 5e-3, hidden_size = 512, balance_factor = 1,
                                      amount_of_patch = 9, moving_average_decay = 0.996,
                                      optimizer_name = 'adam')


device = training_configuration.device

# define data loaders 
"""
slice for debuging
"""

# train_loader, val_loader, test_loader, debug_loader = \
#     initialize_dataloaders(train_df, test_df, 
#                            training_configuration, 
#                            val_split=val_split,  
#                            debug_batch_size = 8, 
#                            random_state = seed,
#                            tb_writer = tb_writer,
#                            train_data=train_data,
#                            test_data=test_data)
# # print size of data-sets
# # print size of data-sets
# print(f'Train length = {train_loader.dataset.data_df.shape[0]}, val length = {val_loader.dataset.data_df.shape[0]}, test length = {test_loader.dataset.data_df.shape[0]}')




# # set model 
model = CNN(training_configuration, 
              num_classes = amount_of_class,
              image_dim = (3,image_dim, image_dim),
              freeze_all=False,
              weights='IMAGENET1K_V1',
              model_name = 'resnet50'
              )

student = None

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










 