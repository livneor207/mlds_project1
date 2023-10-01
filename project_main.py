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
# from albumentations import Compose, Normalize, RandomCrop, Horizont1alFlip, ShiftScaleRotate, HueSaturationValue
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
import argparse

# rnn = nn.LSTM(5, 20, 2) # input size, hidden size, amount of LSTM units
# input = torch.randn(7, 3, 5) # (batch size, amount of features, input history)
# h0 = torch.randn(2, 3, 20) # (amount of LSTM layers,  amount of features, hidden size) 
# # logical that hidden and cell mats will not contain history but summary of history using seq features
# c0 = torch.randn(2, 3, 20) # (amount of LSTM layers,  amount of features, hidden size)
# output, (hn, cn) = rnn(input, (h0, c0)) # (batch size,  amount of features, hidden size)

plt.close('all')
"""
server:~$ python -m spyder_kernels.console - matplotlib=’inline’    --ip=172.17.0.1 -f=./remotemachine.json
python -m spyder_kernels.console --matplotlib='inline' --ip=172.17.0.1

"""

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

# import pandas as pd 
# import glob 
# import os
# csv_folder_path  = r'/home/mlds_user/Desktop/mlds_project1/expirements'
# file_type_to_find =  os.path.join(csv_folder_path, '*.csv')
# all_csv_path = glob.glob(file_type_to_find)
# results_columns = ['sim_name','train_accuracy', 'train_f_score', 'val_accuracy', 'val_f_score_loss']
# result_collector = []
# for i_path in  all_csv_path:
#     i_df =  pd.read_csv(i_path)
#     sim_name = os.path.basename(i_path)
#     i_df_columns = i_df.columns.tolist()
#     if 'val_f1_permutation_score' in i_df_columns:
#         continue
#     else:
#         train_accuracy = i_df['train_accuracy'].max()
#         train_f_score = i_df['train_f_score'].max()
#         val_accuracy = i_df['val_accuracy'].max()
#         val_f_score = i_df['val_f_score_loss'].max()
#         i_results = [sim_name, train_accuracy, train_f_score, val_accuracy, val_f_score]
#         result_collector.append(i_results)
# simulation_summary = pd.DataFrame(result_collector, columns = results_columns)    
    
# simulation_summary
# substring = '_withperm'
# filtered_df = simulation_summary[simulation_summary['sim_name'].str.contains(substring, case=False)]
# substring = '_withoutperm'
# filtered_df = simulation_summary[simulation_summary['sim_name'].str.contains(substring, case=False)]



# Create an Argumenttraining_configuration object
training_configuration = argparse.ArgumentParser(description='Simulation argument')
# Add arguments to the parser


"""
options
task_name  = 'cat_dogs'
task_name  = 'CIFAR10'
task_name  = 'OxfordIIITPet'
"""

"""
!todo 
max_allowed_permutation change
"""
# path
training_configuration.add_argument('--task_name', type=str, default = 'CIFAR100', help='Specify an task to work on')
training_configuration.add_argument('--ssl_model_name', type=str, default='ssl_model', help='Specify a sll model path')
training_configuration.add_argument('--sup_ssl_model_withperm_name', type=str, default='sup_ssl_model_withperm', help='Specify model name for supervsied learning on data with permutation')
training_configuration.add_argument('--sup_ssl_model_withoutperm_name', type=str, default='sup_ssl_model_withoutperm', help='Specify model name for supervsied learning on data without permutation')
training_configuration.add_argument('--sup_model_withoutperm_name', type=str, default='sup_model_withoutperm', help='Specify model name for supervsied learning on data without permutation')
training_configuration.add_argument('--sup_model_withperm_name', type=str, default='sup_model_withperm', help='Specify model name for supervsied learning on data without permutation')

# byol
training_configuration.add_argument('--moving_average_decay', type=float, default = 0.996, help='Specify an factor of how to update target model, shold be greater the 0.9')

# permutation
training_configuration.add_argument('--max_allowed_permutation', type=int, default = 100, help='Specify the amount of allowed permutation from all permutation, should be smaller than 1000')
training_configuration.add_argument('--use_auto_weight', type=int, default = 1, help='Specify if model require to auto adjust the loss coeficient')
training_configuration.add_argument('--amount_of_patch', type=int, default = 4, help='Specify the grid size for permutation defenition')
training_configuration.add_argument('--perm', type=str, default = 'perm', help='Specify use or not permutation augmentation')
training_configuration.add_argument('--balance_factor', type=float, default = 1, help='Specify an factor to postion head prediction loss, if set to 0, remove the PE head')
training_configuration.add_argument('--balance_factor2', type=float, default = 1, help='Specify an factor to permutation index prediction loss, if set to 0, remove the classification head')

# datapreperation
training_configuration.add_argument('--val_split', type=float, default = 0.1, help='Specify validation size')
training_configuration.add_argument('--image_dim', type=int, default = 224, help='Specify image size')
training_configuration.add_argument('--train_split', type=float, default = 1, help='Specify amount of trainig data to be trained')
training_configuration.add_argument('--rand_choise', type=int, default = 1, help='Specify use or not augmentation')
training_configuration.add_argument('--pin_memory', type=int, default=1, help='Specify classification loss name')

# systems
training_configuration.add_argument('--num_workers', type=int, default = 0, help='Specify the amount of worker for dataloader multiprocessing')
training_configuration.add_argument('--seed', type=int, default = 42, help='Specify random state')
training_configuration.add_argument('--googledrive', type=int, default=0, help='Specify classification loss name')

# training
training_configuration.add_argument('--batch_size', type=int, default = 256, help='Specify an batch size for training sould, ssl improve as batch size increases')
training_configuration.add_argument('--max_opt', type=int, default = 0, help='Specify if optimization goal is maximization of minimuzation')
training_configuration.add_argument('--epochs_count', type=int, default = 200, help='Specify the amount of apoch for optimization training')
training_configuration.add_argument('--learning_type', type=str, default = 'self_supervised', help='Specify optimization type between ssl to supervised learning')
training_configuration.add_argument('--ssl_training', type=int, default=1, help='Specify classification loss name')
training_configuration.add_argument('--sup_ssl_withperm', type=int, default=1, help='Specify classification loss name')
training_configuration.add_argument('--sup_ssl_withoutperm', type=int, default =1, help='Specify classification loss name')
training_configuration.add_argument('--sup_withoutperm', type=int, default=1, help='Specify classification loss name')
training_configuration.add_argument('--sup_withperm', type=int, default=1, help='Specify classification loss name')

# loss
training_configuration.add_argument('--loss_functions_name', type=str, default = 'ce', help='Specify final projection size')
training_configuration.add_argument('--classification_loss_name', type=str, default = 'ce', help='Specify classification loss name')
training_configuration.add_argument('--ranking_loss', type=str, default = 'CosineSimilarity', help='Specify classification loss name')
training_configuration.add_argument('--representation_loss', type=str, default = 'CosineSimilarity', help='Specify classification loss name')

# optimizer 
training_configuration.add_argument('--optimizer_name', type=str, default = 'AdamW', help='Specify optimizer name between adam and lion')
training_configuration.add_argument('--weight_decay', type=float, default = 4e-4, help='Specify if to use weight decay regurelaization in optimizer')
training_configuration.add_argument('--learning_rate', type=float, default = 1e-4, help='Specify learning rate for optimization')
training_configuration.add_argument('--scheduler_name', type=str, default = 'None', help='Specify scheduler for optimization')


training_configuration.add_argument('--optimizer_name_ssl', type=str, default = 'AdamW', help='Specify optimizer name between adam and lion')
training_configuration.add_argument('--weight_decay_ssl', type=float, default = 4e-4, help='Specify if to use weight decay regurelaization in optimizer')
training_configuration.add_argument('--learning_rate_ssl', type=float, default = 1e-4, help='Specify learning rate for optimization')

# training_configuration.add_argument('--optimizer_name_ssl', type=str, default = 'Lars', help='Specify optimizer name between adam and lion')
# training_configuration.add_argument('--weight_decay_ssl', type=float, default = 1e-4, help='Specify if to use weight decay regurelaization in optimizer')
# training_configuration.add_argument('--learning_rate_ssl', type=float, default = 2e-1, help='Specify learning rate for optimization')

# model
training_configuration.add_argument('--hidden_size', type=int, default = 512, help='Specify final projection size')
training_configuration.add_argument('--unfreeze', type=int, default=0, help='Specify classification loss name')
training_configuration.add_argument('--copy_weights', type=int, default=0, help='Specify classification loss name')
training_configuration.add_argument('--update_student', type=int, default=1, help='Specify classification loss name')
training_configuration.add_argument('--load_ssl', type=int, default=0, help='Specify classification loss name')
training_configuration.add_argument('--model_layer', type=int, default=7, help='Specify classification loss name')
training_configuration.add_argument('--model_sub_layer', type=int, default=0, help='Specify classification loss name')
training_configuration.add_argument('--pe_dim', type=int, default=512, help='Specify classification loss name')
training_configuration.add_argument('--worm_up', type=int, default=-1, help='Specify classification loss name')



# training_configuration.classification_loss_name = 'ce'
# Parse the command-line arguments
training_configuration = training_configuration.parse_args()

# run validation on trainig confiugration parameters
argparser_validation(training_configuration)

# seed
seed =  training_configuration.seed
val_split = training_configuration.val_split 

image_dim = training_configuration.image_dim
train_split = training_configuration.train_split
rand_choise = training_configuration.rand_choise
debug=  False
if debug: 
    train_split = 0.01
    val_split = 0.001
    training_configuration.batch_size = 8
    training_configuration.epochs_count = 100
    
    training_configuration.ssl_training = 0
    training_configuration.sup_ssl_withperm = 0
    training_configuration.sup_ssl_withoutperm = 0
    training_configuration.sup_withoutperm = 1
    training_configuration.sup_withperm = 0
    # training_configuration.unfreeze = 0
    

seed_everything(seed)

# print(training_configuration.ssl_model_name)
# print(training_configuration.ssl_training)
# print(training_configuration.sup_ssl_withperm)
# print(training_configuration.sup_ssl_withoutperm)
# print(training_configuration.sup_withperm)
# print(training_configuration.sup_withoutperm)
# print(training_configuration.rand_choise)


	

# set path's
if not training_configuration.googledrive:
    current_folder = os.getcwd()
else:
    current_folder = r'/content/gdrive/MyDrive/MLDS_project'
    
    
data_folder = os.path.join(current_folder,  'expirements')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
test_folder_path = os.path.join(data_folder,  'test1', 'test1')
train_folder_path = os.path.join(data_folder,  'train', 'train')

task_name  = training_configuration.task_name

if task_name in ['CIFAR10','CIFAR100', 'cat_dogs']:
    train_df, train_data= parse_train_data(task_name  =task_name, folder_path =train_folder_path, train=True, current_folder= current_folder)
    test_df, test_data = parse_train_data(task_name=task_name, folder_path =test_folder_path, train=False, current_folder = current_folder)
elif task_name in ['OxfordIIITPet', 'FOOD101']:
    train_df, train_data= parse_train_data(task_name  =task_name, folder_path =train_folder_path, train='trainval', current_folder= current_folder)
    test_df, test_data = parse_train_data(task_name=task_name, folder_path =test_folder_path, train='test', current_folder = current_folder)
# parse train data

# 'trainval'
# 'test'
# log to tensorboard, tensorboard summary writer


# get data statistics
train_statistic_df, alpha = data_statistics(train_df)

# get statistic of data
class_ratios, amount_of_class = get_statistic_from_stistic_dataframe(train_statistic_df)

# generate histogram to illustate if data is balance or not 
generate_hitogram_base_dataframe_column(train_df, 'class_name')

# get device
device = training_configuration.device
unfreeze = training_configuration.unfreeze
pin_memory = training_configuration.pin_memory
copy_weights = training_configuration.copy_weights
load_ssl = training_configuration.load_ssl
update_student = training_configuration.update_student
######### start ssl leanring ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if training_configuration.ssl_training:
    
    sim_name = training_configuration.ssl_model_name
    print('simulation '  +sim_name + ' has been started' )
    model_path = os.path.join(data_folder,  sim_name + '.pth')
    log_dir = os.path.join(data_folder,  sim_name)

    log_dir = tb_writer = SummaryWriter(
        log_dir = log_dir,
        comment = "TensorBoard in PyTorch")
    
    
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
                                rand_choise = rand_choise,
                                train_split = train_split,
                                pin_memory = pin_memory)
        
    # print size of data-sets
    print(f'Train length = {train_loader.dataset.data_df.shape[0]}, \
          val length = {val_loader.dataset.data_df.shape[0]},  \
          test length = {test_loader.dataset.data_df.shape[0]}')
    
    
    # generate model
    model = CNN(training_configuration,
                  num_classes = amount_of_class,
                  image_dim = (3,image_dim, image_dim),
                  freeze_all = False,
                  model_name = 'resnet50',
                  weights = 'IMAGENET1K_V1',
                  unfreeze = unfreeze)
    
    if load_ssl:
        # model.load_state_dict(torch.load(model_path))
        # model = torch.load(model_path)
        model = torch.load(model_path, map_location='cpu')
        for name, param in model.named_parameters():
            if param.device.type != 'cpu':
                param.to('cpu')

        
    student = generate_student(model, 
                                training_configuration, 
                                image_dim, 
                                amount_of_class,
                                model_name = 'resnet50',
                                weights = None,
                                unfreeze = unfreeze,
                                copy_weights = copy_weights,
                                update_student = update_student)
    
    
    model.to(device)
    student.to(device)
    summary(model, (3,image_dim, image_dim))
    if not student is None:
        summary(student, (3,image_dim, image_dim))
    
    # set optimizer 
    optimizer, scheduler, scheduler_worm_up =  set_optimizer(model, training_configuration, train_loader, amount_of_class = amount_of_class, alpha = alpha)
    
    # set accuracy metrics
    accuracy_metric  = set_metric(training_configuration, amount_of_class = amount_of_class, metric_name = 'accuracy')
    f_score_accuracy_metric  = set_metric(training_configuration, amount_of_class = amount_of_class, metric_name = 'f_score')
    
    # set loss functions
    if training_configuration.learning_type == 'supervised':
        criterion =  set_classifcation_loss(training_configuration, alpha = alpha)
    else:    
        criterion=  set_similiarities_loss(classification_loss_name = training_configuration.representation_loss, beta = 1)
    
    ranking_criterion = set_rank_loss(loss_name = training_configuration.ranking_loss, margin = 1, num_labels = 1, beta = 1)
    perm_creterion = nn.CrossEntropyLoss()
    
    # show example for data after transformations    
    # generate data generation example
    image, label, perm_order, class_name, perm_label = generate_input_generation_examples(debug_loader)
    # torch.save(model, model_path)

    # im1 = image[0:3,0:3,:,:]
    # im2 = image[0:3,3::,:,:]
    # representation_pred_1, perm_pred_1, perm_label_pred_1 =  model(im1)
    # representation_pred_2, perm_pred_2, perm_label_pred_2 = student(im2)
    # representation_pred_2 = torch.flip(representation_pred_2, dims=[0])    
    # ranking_criterion(representation_pred_1, representation_pred_2)
    
    train_results_df = main(model, student, optimizer, criterion,
                            ranking_criterion, accuracy_metric , perm_creterion,
                            train_loader, val_loader,
                            num_epochs=training_configuration.epochs_count,
                            device=device, 
                            tb_writer=tb_writer, 
                            max_opt = training_configuration.max_opt, 
                            model_path = model_path, 
                            scheduler = scheduler, scheduler_worm_up=scheduler_worm_up)



######### end ssl leanring ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
######## start few shot learning with perm ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if training_configuration.sup_ssl_withperm:
    
    sim_name = training_configuration.sup_ssl_model_withperm_name
    print('simulation '  +sim_name + ' has been started' )
    
    
    model_load_path =  os.path.join(data_folder, training_configuration.ssl_model_name  + '.pth')
    model_path = os.path.join(data_folder, sim_name  + '.pth')
    
    log_dir = os.path.join(data_folder,  training_configuration.sup_ssl_model_withperm_name)

    log_dir = tb_writer = SummaryWriter(
        log_dir = log_dir,
        comment = "TensorBoard in PyTorch")
    
    
    # training_configuration.get_device_type()
    
    training_configuration.perm = 'perm'
    training_configuration.weight_decay = 0 
    training_configuration.max_opt = True
    training_configuration.learning_type = 'self_supervised'
    
    device = training_configuration.device
    
    model = CNN(training_configuration,
                 num_classes = amount_of_class,
                 image_dim = (3,image_dim, image_dim),
                freeze_all = False,
                model_name = 'resnet50',
                weights = 'IMAGENET1K_V1',
                unfreeze = unfreeze)
    
    
    
    training_configuration.learning_type = 'supervised'
    model = torch.load(model_load_path, map_location='cpu')
    for name, param in model.named_parameters():
        if param.device.type != 'cpu':
            param.to('cpu')
    # model = torch.load(model_load_path)
    # model = model.to('cpu')
    # model.load_state_dict(torch.load(model_load_path))
    print('model sigma')
    print(model.sigma)
    # model_path = os.path.join(data_folder,  'model2.pth')
    
    ssl_model =  SSLMODEL(model,
                          num_classes=amount_of_class,
                          image_dim=(3,image_dim, image_dim),
                          freeze_all = False,
                          model_name = 'resnet50')
    student= None
    ssl_model.learning_type = 'supervised'
    ssl_model = ssl_model.to(device)
    summary(ssl_model, (3,image_dim, image_dim))
    
    
    
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
                                rand_choise = rand_choise,
                                train_split = train_split,
                                pin_memory = pin_memory)
        
        
    # print size of data-sets
    print(f'Train length = {train_loader.dataset.data_df.shape[0]}, val length = {val_loader.dataset.data_df.shape[0]}, test length = {test_loader.dataset.data_df.shape[0]}')
    
    # set optimizer
    optimizer, scheduler, scheduler_worm_up =  set_optimizer(ssl_model, training_configuration, train_loader, amount_of_class = amount_of_class, alpha = alpha)
    
    # set accuracy metrics
    accuracy_metric  = set_metric(training_configuration, amount_of_class = amount_of_class, metric_name = 'accuracy')
    f_score_accuracy_metric  = set_metric(training_configuration, amount_of_class = amount_of_class, metric_name = 'f_score')
    
    # set loss functions
    if training_configuration.learning_type == 'supervised':
        criterion =  set_classifcation_loss(training_configuration, alpha = alpha)
    else:
        criterion=  set_similiarities_loss(classification_loss_name = training_configuration.representation_loss, beta = 1)
    ranking_criterion = set_rank_loss(loss_name = training_configuration.ranking_loss, margin = 1, num_labels = 1, beta = 1)
    perm_creterion = nn.CrossEntropyLoss()
    
    
    
    # show example for data after transformations
    # generate data generation example
    image, label, perm_order, class_name, perm_label = generate_input_generation_examples(debug_loader)
    
    
    train_results_df = main(ssl_model, student, optimizer, criterion, ranking_criterion, accuracy_metric , perm_creterion,
                            train_loader, val_loader, num_epochs=training_configuration.epochs_count, device=device, 
                            tb_writer=tb_writer, max_opt = training_configuration.max_opt, model_path = model_path, 
                            scheduler = scheduler)



# # ######### end few shot learning with perm ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # ######### start few shot learning without perm ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if training_configuration.sup_ssl_withoutperm:

    sim_name = training_configuration.sup_ssl_model_withoutperm_name
    print('simulation '  +sim_name + ' has been started' )
    # """
    # fuuly - supervised
    # """
    
    model_load_path =  os.path.join(data_folder, training_configuration.ssl_model_name  + '.pth')
    model_path = os.path.join(data_folder, sim_name  + '.pth')
    
    log_dir = os.path.join(data_folder,  training_configuration.sup_ssl_model_withoutperm_name)

    log_dir = tb_writer = SummaryWriter(
        log_dir = log_dir,
        comment = "TensorBoard in PyTorch")
    
    
    training_configuration.weight_decay = 0 
    training_configuration.perm = 'no_perm'
    training_configuration.max_opt = True
    training_configuration.learning_type = 'self_supervised'
    
    # training_configuration.update_merics(loss_functions_name = 'ce', learning_rate = 1e-3,
    #                                       learning_type='supervised', batch_size= 16, 
    #                                       scheduler_name = 'None', max_opt = True,
    #                                       epochs_count = 50, perm= 'perm', num_workers = 0, 
    #                                       max_lr = 5e-3, hidden_size = 512, balance_factor = 1,
    #                                       amount_of_patch = 9, moving_average_decay = 0.996,
    #                                       optimizer_name = 'adam')
    
    
    device = training_configuration.device
    
    # define data loaders 
    """
    slice for debuging
    """
    
    
    # set model
    # set model
    
    model = CNN(training_configuration,
                 num_classes = amount_of_class,
                 image_dim = (3,image_dim, image_dim),
                 freeze_all = False,
                 model_name = 'resnet50',
                 weights = 'IMAGENET1K_V1',
                 unfreeze = unfreeze)
    
    
    student= None
    training_configuration.learning_type = 'supervised'
    # model.load_state_dict(torch.load(model_load_path))
    # model = torch.load(model_load_path)
    model = torch.load(model_load_path, map_location='cpu')
    for name, param in model.named_parameters():
        if param.device.type != 'cpu':
            param.to('cpu')

    print('model sigma')
    print(model.sigma)
    
    ssl_model =  SSLMODEL(model,
                          num_classes=amount_of_class,
                          image_dim=(3,image_dim, image_dim),
                          freeze_all = False,
                          model_name = 'resnet50')
    ssl_model.to(device)
    student= None
    summary(ssl_model, (3,image_dim, image_dim))

    
    # define  loaders 
    
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
                                rand_choise = rand_choise,
                                train_split = train_split,
                                pin_memory = pin_memory)
        
    
    
    
    
    # set optimizer
    optimizer, scheduler, scheduler_worm_up =  set_optimizer(ssl_model, training_configuration, train_loader, amount_of_class = amount_of_class, alpha = alpha)
    
    # set accuracy metrics
    accuracy_metric  = set_metric(training_configuration, amount_of_class = amount_of_class, metric_name = 'accuracy')
    f_score_accuracy_metric  = set_metric(training_configuration, amount_of_class = amount_of_class, metric_name = 'f_score')
    
    # set loss functions
    if training_configuration.learning_type == 'supervised':
        criterion =  set_classifcation_loss(training_configuration, alpha = alpha)
    else:
        criterion=  set_similiarities_loss(classification_loss_name = training_configuration.representation_loss, beta = 1)
    ranking_criterion = set_rank_loss(loss_name = training_configuration.ranking_loss, margin = 1, num_labels = 1, beta = 1)
    perm_creterion = nn.CrossEntropyLoss()
    
    # show example for data after transformations
    # generate data generation example
    image, label, perm_order, class_name, perm_label = generate_input_generation_examples(debug_loader)
    
    
    
    gc.collect()
    
    train_results_df = main(ssl_model, student, optimizer, criterion, ranking_criterion, accuracy_metric , perm_creterion,
                            train_loader, val_loader, num_epochs=training_configuration.epochs_count, device=device,
                            tb_writer=tb_writer, max_opt = training_configuration.max_opt, 
                            model_path = model_path, scheduler = scheduler)



if training_configuration.sup_withoutperm:

    # """
    # fuuly - supervised
    # """
    sim_name = training_configuration.sup_model_withoutperm_name
    print('simulation '  +sim_name + ' has been started' )
    
    model_path = os.path.join(data_folder,  sim_name + '.pth')

    log_dir = os.path.join(data_folder,  sim_name)

    log_dir = tb_writer = SummaryWriter(
        log_dir = log_dir,
        comment = "TensorBoard in PyTorch")
    
    
    training_configuration.weight_decay = 0 
    training_configuration.perm = 'no_perm'
    training_configuration.max_opt = True
    training_configuration.learning_type = 'supervised'
    
    # training_configuration.update_merics(loss_functions_name = 'ce', learning_rate = 1e-3,
    #                                       learning_type='supervised', batch_size= 16, 
    #                                       scheduler_name = 'None', max_opt = True,
    #                                       epochs_count = 50, perm= 'perm', num_workers = 0, 
    #                                       max_lr = 5e-3, hidden_size = 512, balance_factor = 1,
    #                                       amount_of_patch = 9, moving_average_decay = 0.996,
    #                                       optimizer_name = 'adam')
    
    
    device = training_configuration.device
    
    # define data loaders 
    """
    slice for debuging
    """
    
    
    # set model
    # set model
    
    model = CNN(training_configuration,
                 num_classes = amount_of_class,
                 image_dim = (3,image_dim, image_dim),
                 freeze_all = False,
                 model_name = 'resnet50',
                 weights = 'IMAGENET1K_V1',
                 unfreeze = unfreeze)
    
    
    student= None
    
  
    model.to(device)
    summary(model, (3,image_dim, image_dim))

    # define  loaders 
    
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
                                rand_choise = rand_choise,
                                train_split = train_split,
                                pin_memory = pin_memory)
    
    
    
    
    # set optimizer
    optimizer, scheduler, scheduler_worm_up =  set_optimizer(model, training_configuration, train_loader, amount_of_class = amount_of_class, alpha = alpha)
    
    # set accuracy metrics
    accuracy_metric  = set_metric(training_configuration, amount_of_class = amount_of_class, metric_name = 'accuracy')
    f_score_accuracy_metric  = set_metric(training_configuration, amount_of_class = amount_of_class, metric_name = 'f_score')
    
    # set loss functions
    if training_configuration.learning_type == 'supervised':
        criterion =  set_classifcation_loss(training_configuration, alpha = alpha)
    else:
        criterion=  set_similiarities_loss(classification_loss_name = training_configuration.representation_loss, beta = 1)
    ranking_criterion = set_rank_loss(loss_name = training_configuration.ranking_loss, margin = 1, num_labels = 1, beta = 1)
    perm_creterion = nn.CrossEntropyLoss()
    



    # show example for data after transformations
    # generate data generation example
    image, label, perm_order, class_name, perm_label = generate_input_generation_examples(debug_loader)
    
    
    
    gc.collect()
    
    train_results_df = main(model, student, optimizer, criterion, ranking_criterion, accuracy_metric , perm_creterion,
                            train_loader, val_loader, num_epochs=training_configuration.epochs_count, device=device,
                            tb_writer=tb_writer, max_opt = training_configuration.max_opt, 
                            model_path = model_path, scheduler = scheduler, 
                            scheduler_worm_up= scheduler_worm_up)



# # ######### end few shot learning with perm ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # ######### start few shot learning without perm ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if training_configuration.sup_withperm:


    # """
    # fuuly - supervised
    # """
    sim_name = training_configuration.sup_model_withperm_name
    print('simulation '  +sim_name + ' has been started' )
    model_path = os.path.join(data_folder, sim_name  + '.pth')
    log_dir = os.path.join(data_folder,  sim_name)
    
    log_dir = tb_writer = SummaryWriter(
        log_dir = log_dir,
        comment = "TensorBoard in PyTorch")
    
    
    training_configuration.weight_decay = 0 
    training_configuration.perm = 'perm'
    training_configuration.max_opt = True
    training_configuration.learning_type = 'supervised'
    
    # training_configuration.update_merics(loss_functions_name = 'ce', learning_rate = 1e-3,
    #                                       learning_type='supervised', batch_size= 16, 
    #                                       scheduler_name = 'None', max_opt = True,
    #                                       epochs_count = 50, perm= 'perm', num_workers = 0, 
    #                                       max_lr = 5e-3, hidden_size = 512, balance_factor = 1,
    #                                       amount_of_patch = 9, moving_average_decay = 0.996,
    #                                       optimizer_name = 'adam')
    
    
    device = training_configuration.device
    
    # define data loaders 
    """
    slice for debuging
    """
    
    
    # set model
    # set model
    
    model = CNN(training_configuration,
                 num_classes = amount_of_class,
                 image_dim = (3,image_dim, image_dim),
                 freeze_all = False,
                 model_name = 'resnet50',
                 weights = 'IMAGENET1K_V1',
                 unfreeze = unfreeze)
    
    
    student= None
    
  
    model.to(device)
    summary(model, (3,image_dim, image_dim))

    # define  loaders 
    
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
                                rand_choise = rand_choise,
                                train_split = train_split,
                                pin_memory = pin_memory)
        
    
    
    
    
    # set optimizer
    optimizer, scheduler, scheduler_worm_up =  set_optimizer(model, training_configuration, train_loader, amount_of_class = amount_of_class, alpha = alpha)
    
    # set accuracy metrics
    accuracy_metric  = set_metric(training_configuration, amount_of_class = amount_of_class, metric_name = 'accuracy')
    f_score_accuracy_metric  = set_metric(training_configuration, amount_of_class = amount_of_class, metric_name = 'f_score')
    
    # set loss functions
    if training_configuration.learning_type == 'supervised':
        criterion =  set_classifcation_loss(training_configuration, alpha = alpha)
    else:
        criterion=  set_similiarities_loss(classification_loss_name = training_configuration.representation_loss, beta = 1)
    ranking_criterion = set_rank_loss(loss_name = training_configuration.ranking_loss, margin = 1, num_labels = 1, beta = 1)
    perm_creterion = nn.CrossEntropyLoss()
    
    # show example for data after transformations
    # generate data generation example
    image, label, perm_order, class_name, perm_label = generate_input_generation_examples(debug_loader)
    
    
    
    gc.collect()
    
    train_results_df = main(model, student, optimizer, criterion, ranking_criterion, accuracy_metric , perm_creterion,
                            train_loader, val_loader, num_epochs=training_configuration.epochs_count, device=device,
                            tb_writer=tb_writer, max_opt = training_configuration.max_opt, 
                            model_path = model_path, scheduler = scheduler, scheduler_worm_up=scheduler_worm_up)















 