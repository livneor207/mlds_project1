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


plt.close('all')

"""
dash board tensorboard 
tensorboard --logdir logdir_folder_path --port default
tensorboard --logdir "C:\MSC\opencv-python-free-course-code\classification_project\opencv-pytorch-dl-course-classification\expirement1" --port default
"""
# Create an Argumenttraining_configuration object
training_configuration = argparse.ArgumentParser(description='Simulation argument')

# path
training_configuration.add_argument('--task_name', type=str, default = 'CIFAR10', help='Specify an task to work on (CIFAR10\CIFAR100\cat_dogs\OxfordIIITPet\FOOD101)')
training_configuration.add_argument('--ssl_model_name', type=str, default='ssl_model', help='Specify a sll model file name')
training_configuration.add_argument('--sup_ssl_model_withperm_name', type=str, default='sup_ssl_model_withperm', help='Specify model name for supervsied learning on data with permutation')
training_configuration.add_argument('--sup_ssl_model_withoutperm_name', type=str, default='sup_ssl_model_withoutperm', help='Specify model name for supervsied learning on data without permutation')
training_configuration.add_argument('--sup_model_withoutperm_name', type=str, default='sup_model_withoutperm', help='Specify model name for supervsied learning on data without permutation')
training_configuration.add_argument('--sup_model_withperm_name', type=str, default='sup_model_withperm', help='Specify model name for supervsied learning on data without permutation')

# byol
training_configuration.add_argument('--moving_average_decay', type=float, default = 0.996, help='Specify an factor of how to update target model, shold be greater the 0.9')

# permutation
training_configuration.add_argument('--max_allowed_permutation', type=int, default = 24, help='Specify the amount of allowed permutation from all permutation, should be smaller than 1000')
training_configuration.add_argument('--use_auto_weight', type=int, default = 1, help='Specify if model require to auto adjust the loss coeficient')
training_configuration.add_argument('--amount_of_patch', type=int, default = 4, help='Specify the grid size for permutation defenition')
training_configuration.add_argument('--perm', type=str, default = 'perm', help='Specify use or not permutation augmentation')
training_configuration.add_argument('--balance_factor', type=float, default = 1, help='Specify an factor to postion head prediction loss, if set to 0, remove the PE head')
training_configuration.add_argument('--balance_factor2', type=float, default = 1, help='Specify an factor to permutation index prediction loss, if set to 0, remove the classification head')

# datapreperation
training_configuration.add_argument('--val_split', type=float, default = 0.2, help='Specify validation size')
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
training_configuration.add_argument('--ssl_training', type=int, default=1, help='Specify run ssl training')
training_configuration.add_argument('--sup_ssl_withperm', type=int, default=1, help='Specify supervised training, data with permutation, backbone initilized using ssl model')
training_configuration.add_argument('--sup_ssl_withoutperm', type=int, default =1, help='Specify supervised training, data without permutation, backbone initilized using ssl model')
training_configuration.add_argument('--sup_withoutperm', type=int, default=1, help='Specify supervised training, data without permutation, backbone initilized ImageNet')
training_configuration.add_argument('--sup_withperm', type=int, default=1, help='Specify supervised training, data with permutation, backbone initilized ImageNet')
training_configuration.add_argument('--train_model', type=int, default=1, help='Specify if to run supervised learning')


# loss
training_configuration.add_argument('--loss_functions_name', type=str, default = 'ce', help='Specify loss function')
training_configuration.add_argument('--classification_loss_name', type=str, default = 'ce', help='Specify classification loss name')
training_configuration.add_argument('--ranking_loss', type=str, default = 'CosineSimilarity', help='Specify postion embedding loss name')
training_configuration.add_argument('--representation_loss', type=str, default = 'CosineSimilarity', help='Specify representation loss name')

# optimizer 
training_configuration.add_argument('--optimizer_name', type=str, default = 'AdamW', help='Specify optimizer name between adam and lion')
training_configuration.add_argument('--weight_decay', type=float, default = 4e-4, help='Specify if to use weight decay regurelaization in optimizer')
training_configuration.add_argument('--learning_rate', type=float, default = 1e-4, help='Specify learning rate for optimization')
training_configuration.add_argument('--scheduler_name', type=str, default = 'None', help='Specify scheduler for optimization')

# training_configuration.add_argument('--optimizer_name_ssl', type=str, default = 'AdamW', help='Specify optimizer name between adam and lion')
# training_configuration.add_argument('--weight_decay_ssl', type=float, default = 4e-4, help='Specify if to use weight decay regurelaization in optimizer')
# training_configuration.add_argument('--learning_rate_ssl', type=float, default = 1e-4, help='Specify learning rate for optimization')

training_configuration.add_argument('--optimizer_name_ssl', type=str, default = 'Lars', help='Specify optimizer name between adam and lion')
training_configuration.add_argument('--weight_decay_ssl', type=float, default = 4e-4, help='Specify if to use weight decay regurelaization in optimizer')
training_configuration.add_argument('--learning_rate_ssl', type=float, default = 2e-1, help='Specify learning rate for optimization')

# model
training_configuration.add_argument('--hidden_size', type=int, default = 512, help='Specify prediction layer size')
training_configuration.add_argument('--unfreeze', type=int, default=0, help='train all model parameters')
training_configuration.add_argument('--copy_weights', type=int, default=0, help='copy model wieght to student ')
training_configuration.add_argument('--update_student', type=int, default=1, help='copy weight from model to student when generate the student ')
training_configuration.add_argument('--load_ssl', type=int, default=0, help='Specify if to start from last trained ssl model')
training_configuration.add_argument('--model_layer', type=int, default=7, help='Specify from which layer to take output at loop forward')
training_configuration.add_argument('--model_sub_layer', type=int, default=0, help='Specify from which sublayer to take output at loop forward')
training_configuration.add_argument('--pe_dim', type=int, default=512, help='Specify postion embedding vector size ')
training_configuration.add_argument('--worm_up', type=int, default=-1, help='Specify amount of epoch to perfrom worm up')

# Parse the command-line arguments
training_configuration = training_configuration.parse_args()

# run validation on trainig confiugration parameters
argparser_validation(training_configuration)

# seed
seed =  training_configuration.seed
val_split = training_configuration.val_split 

# get data parameters 
image_dim = training_configuration.image_dim
train_split = training_configuration.train_split
rand_choise = training_configuration.rand_choise
debug=  False
if debug: 
    train_split = 0.01
    val_split = 0.001
    training_configuration.batch_size = 8
    training_configuration.epochs_count = 100
    
    training_configuration.ssl_training = 1
    training_configuration.sup_ssl_withperm = 0
    training_configuration.sup_ssl_withoutperm = 1
    training_configuration.sup_withoutperm = 1
    training_configuration.sup_withperm = 0
    training_configuration.train_model = 0
    # training_configuration.unfreeze = 0
    
# seed all
seed_everything(seed)

# set path's
if not training_configuration.googledrive:
    current_folder = os.getcwd()
else:
    current_folder = r'/content/gdrive/MyDrive/MLDS_project'
    
# set expirements folder 
data_folder = os.path.join(current_folder,  'expirements')
# need to generate new folder for expirements 
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
    
# generate path for CatDogs dataset
test_folder_path = os.path.join(data_folder,  'test1', 'test1')
train_folder_path = os.path.join(data_folder,  'train', 'train')

# get dataset name 
task_name  = training_configuration.task_name

# not require to download data
if task_name in ['CIFAR10','CIFAR100', 'cat_dogs']:
    train_df, train_data= parse_train_data(task_name  =task_name, folder_path =train_folder_path, train=True, current_folder= current_folder)
    test_df, test_data = parse_train_data(task_name=task_name, folder_path =test_folder_path, train=False, current_folder = current_folder)
# require to download data
elif task_name in ['OxfordIIITPet', 'FOOD101']:
    train_df, train_data= parse_train_data(task_name  =task_name, folder_path =train_folder_path, train='trainval', current_folder= current_folder)
    test_df, test_data = parse_train_data(task_name=task_name, folder_path =test_folder_path, train='test', current_folder = current_folder)

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
    
    # simulation name 
    sim_name = training_configuration.ssl_model_name
    print('simulation '  +sim_name + ' has been started' )
    
    # path defintion
    model_path = os.path.join(data_folder,  sim_name + '_epochs.pth')
    log_dir = os.path.join(data_folder,  sim_name)
    
    # tensorboard dir 
    log_dir = tb_writer = SummaryWriter(
        log_dir = log_dir,
        comment = "TensorBoard in PyTorch")
    
    
    # define data loaders 
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
    
    
    # generate model
    model = CNN(training_configuration,
                  num_classes = amount_of_class,
                  image_dim = (3,image_dim, image_dim),
                  freeze_all = False,
                  model_name = 'resnet50',
                  weights = 'IMAGENET1K_V1',
                  unfreeze = unfreeze)
    
    # start from last traind ssl model 
    if load_ssl:
        model = load_model(model_path, learning_type = training_configuration.learning_type)

    # generate student model
    student = generate_student(model, 
                                training_configuration, 
                                image_dim, 
                                amount_of_class,
                                model_name = 'resnet50',
                                weights = 'IMAGENET1K_V1',
                                unfreeze = unfreeze,
                                copy_weights = copy_weights,
                                update_student = update_student)
    
   
    # send models into device 
    model.to(device)
    student.to(device)
    
    # print model summary 
    summary(model, (3,image_dim, image_dim))
    if not student is None:
        summary(student, (3,image_dim, image_dim))
    
    # set optimizer 
    optimizer, scheduler, scheduler_worm_up =  set_optimizer(model, training_configuration, train_loader, amount_of_class = amount_of_class, alpha = alpha)
    
    # set accuracy metrics
    accuracy_metric  = set_metric(training_configuration, amount_of_class = amount_of_class, metric_name = 'accuracy')
    f_score_accuracy_metric  = set_metric(training_configuration, amount_of_class = amount_of_class, metric_name = 'f_score')
    
    # set loss functions
    # supervised 
    if training_configuration.learning_type == 'supervised':
        criterion =  set_classifcation_loss(training_configuration, alpha = alpha)
    # ssl loss
    else:    
        criterion=  set_similiarities_loss(classification_loss_name = training_configuration.representation_loss, beta = 1)
    
    # set postion embedding loss, and permutation index prediction loss
    ranking_criterion = set_rank_loss(loss_name = training_configuration.ranking_loss, margin = 1, num_labels = 1, beta = 1)
    perm_creterion = nn.CrossEntropyLoss()
    
    # show example for data after transformations    
    # generate data generation example
    image, label, perm_order, class_name, perm_label = generate_input_generation_examples(debug_loader)

    
    # start training 
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
    
    # set model name 
    sim_name = training_configuration.sup_ssl_model_withperm_name
    print('simulation '  +sim_name + ' has been started' )
    
    # define path's
    model_load_path =  os.path.join(data_folder, training_configuration.ssl_model_name  + '.pth')
    model_path = os.path.join(data_folder, sim_name  + '_epochs.pth')
    train_val_test_summary = os.path.join(data_folder, sim_name  + '_summary.csv')
    
    # set tensorboard  
    log_dir = os.path.join(data_folder,  training_configuration.sup_ssl_model_withperm_name)
    log_dir = tb_writer = SummaryWriter(
        log_dir = log_dir,
        comment = "TensorBoard in PyTorch")
    
    
    # predefined parameter for this simulation
    training_configuration.perm = 'perm'
    training_configuration.weight_decay = 1e-4 
    training_configuration.max_opt = True
    training_configuration.learning_type = 'self_supervised'
    device = training_configuration.device
    
    # set model 
    model = CNN(training_configuration,
                 num_classes = amount_of_class,
                 image_dim = (3,image_dim, image_dim),
                freeze_all = False,
                model_name = 'resnet50',
                weights = 'IMAGENET1K_V1',
                unfreeze = unfreeze)
    
    # load ssl model 
    model = load_model(model_load_path, learning_type = training_configuration.learning_type)
    
    # set learning type 
    training_configuration.learning_type = 'supervised'
    
    # convert loaded model into ssl model --> take backbone and add classifier head 
    ssl_model =  SSLMODEL(model,
                          num_classes=amount_of_class,
                          image_dim=(3,image_dim, image_dim),
                          freeze_all = False,
                          model_name = 'resnet50')
    # set supervised defintion
    student= None
    ssl_model.learning_type = 'supervised'
    
    # send ssl model to device 
    ssl_model = ssl_model.to(device)
    
    # print model 
    summary(ssl_model, (3,image_dim, image_dim))
    
    # define data loaders 
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
    
    # set loss 
    ranking_criterion = set_rank_loss(loss_name = training_configuration.ranking_loss, margin = 1, num_labels = 1, beta = 1)
    perm_creterion = nn.CrossEntropyLoss()
    
    
    
    # show example for data after transformations
    # generate data generation example
    image, label, perm_order, class_name, perm_label = generate_input_generation_examples(debug_loader)
    
    # train supervised the ssl model 
    if training_configuration.train_model:
        train_results_df = main(ssl_model, student, optimizer, criterion, ranking_criterion, accuracy_metric , perm_creterion,
                                train_loader, val_loader, num_epochs=training_configuration.epochs_count, device=device, 
                                tb_writer=tb_writer, max_opt = training_configuration.max_opt, model_path = model_path, 
                                scheduler = scheduler)
        
    
    # run model to evaluate preformance 
    summary_modelresult_df = get_model_results(model, student, model_path, criterion,
                                  ranking_criterion, accuracy_metric, perm_creterion,
                                  train_loader,val_loader,test_loader, device,
                                  train_val_test_summary)

# # ######### end few shot learning with perm ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # ######### start few shot learning without perm ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if training_configuration.sup_ssl_withoutperm:
    # set model name 
    sim_name = training_configuration.sup_ssl_model_withoutperm_name
    print('simulation '  +sim_name + ' has been started' )
    
    # set path's
    model_load_path =  os.path.join(data_folder, training_configuration.ssl_model_name  + '.pth')
    model_path = os.path.join(data_folder, sim_name  + '_epochs.pth')
    train_val_test_summary = os.path.join(data_folder, sim_name  + '_summary.csv')

    # tensorboard 
    log_dir = os.path.join(data_folder,  training_configuration.sup_ssl_model_withoutperm_name)
    log_dir = tb_writer = SummaryWriter(
        log_dir = log_dir,
        comment = "TensorBoard in PyTorch")
    
    # set simulation defention const
    training_configuration.weight_decay = 1e-4
    training_configuration.perm = 'no_perm'
    training_configuration.max_opt = True
    training_configuration.learning_type = 'self_supervised'
    
    # get devide 
    device = training_configuration.device
    
    # set model 
    model = CNN(training_configuration,
                 num_classes = amount_of_class,
                 image_dim = (3,image_dim, image_dim),
                 freeze_all = False,
                 model_name = 'resnet50',
                 weights = 'IMAGENET1K_V1',
                 unfreeze = unfreeze)
    
    # set student for supervised which is no necessary    
    student= None
 
    # load ssl model 
    model = load_model(model_load_path, learning_type = training_configuration.learning_type)
    
    # set learning type 
    training_configuration.learning_type = 'supervised'

    # generate model for supervised task using backbone of ssl model
    ssl_model =  SSLMODEL(model,
                          num_classes=amount_of_class,
                          image_dim=(3,image_dim, image_dim),
                          freeze_all = False,
                          model_name = 'resnet50')
    
    # send model to device 
    ssl_model.to(device)
    
    # student 
    student= None
    
    # print summary 
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
    
    
    # send model to train
    gc.collect()
    if training_configuration.train_model:
        train_results_df = main(ssl_model, student, optimizer, criterion, ranking_criterion, accuracy_metric , perm_creterion,
                                train_loader, val_loader, num_epochs=training_configuration.epochs_count, device=device,
                                tb_writer=tb_writer, max_opt = training_configuration.max_opt, 
                                model_path = model_path, scheduler = scheduler)

    
    
    # evaluate model on train\test\val
    summary_modelresult_df = get_model_results(model, student, model_path, criterion,
                                  ranking_criterion, accuracy_metric, perm_creterion,
                                  train_loader,val_loader,test_loader, device,
                                  train_val_test_summary)


if training_configuration.sup_withoutperm:

    # """
    # fuuly - supervised
    # """
    
    # simulation name 
    sim_name = training_configuration.sup_model_withoutperm_name
    print('simulation '  +sim_name + ' has been started' )
    
    # set model path 
    model_path = os.path.join(data_folder,  sim_name + '_epochs.pth')
    train_val_test_summary = os.path.join(data_folder, sim_name  + '_summary.csv')
    
    # tesorboard path
    log_dir = os.path.join(data_folder,  sim_name)
    log_dir = tb_writer = SummaryWriter(
        log_dir = log_dir,
        comment = "TensorBoard in PyTorch")
    
    # set simulation configuration 
    training_configuration.weight_decay = 1e-4
    training_configuration.perm = 'no_perm'
    training_configuration.max_opt = True
    training_configuration.learning_type = 'supervised'
    
    # get device 
    device = training_configuration.device
    
    # define data loaders 
    
    # set model
    model = CNN(training_configuration,
                 num_classes = amount_of_class,
                 image_dim = (3,image_dim, image_dim),
                 freeze_all = False,
                 model_name = 'resnet50',
                 weights = 'IMAGENET1K_V1',
                 unfreeze = unfreeze)
    
    # set student
    student= None
    
    # send model into device 
    model.to(device)
    
    # print summary 
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
    
    # run model training 
    gc.collect()
    if training_configuration.train_model:
        train_results_df = main(model, student, optimizer, criterion, ranking_criterion, accuracy_metric , perm_creterion,
                                train_loader, val_loader, num_epochs=training_configuration.epochs_count, device=device,
                                tb_writer=tb_writer, max_opt = training_configuration.max_opt, 
                                model_path = model_path, scheduler = scheduler, 
                                scheduler_worm_up= scheduler_worm_up)
        
    # EVALUATE MODEL
    summary_modelresult_df = get_model_results(model, student, model_path, criterion,
                                  ranking_criterion, accuracy_metric, perm_creterion,
                                  train_loader,val_loader,test_loader, device,
                                  train_val_test_summary)
    


# # ######### end few shot learning with perm ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # ######### start few shot learning without perm ##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if training_configuration.sup_withperm:


    # """
    # fuuly - supervised
    # """
    
    # set model name 
    sim_name = training_configuration.sup_model_withperm_name
    print('simulation '  +sim_name + ' has been started' )
    
    # set path's
    model_path = os.path.join(data_folder, sim_name  + '_epochs')
    log_dir = os.path.join(data_folder,  sim_name)
    train_val_test_summary = os.path.join(data_folder, sim_name  + '_summary.csv')
    
    
    # tesorboard path
    log_dir = tb_writer = SummaryWriter(
        log_dir = log_dir,
        comment = "TensorBoard in PyTorch")
    
    # simulation configuration
    training_configuration.weight_decay = 1e-4
    training_configuration.perm = 'perm'
    training_configuration.max_opt = True
    training_configuration.learning_type = 'supervised'
    
    # get device 
    device = training_configuration.device
    
    # set model
    model = CNN(training_configuration,
                 num_classes = amount_of_class,
                 image_dim = (3,image_dim, image_dim),
                 freeze_all = False,
                 model_name = 'resnet50',
                 weights = 'IMAGENET1K_V1',
                 unfreeze = unfreeze)
    
    # set student 
    student= None
    
    # send model to device 
    model.to(device)
    
    # print summary of model 
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
    optimizer, scheduler, scheduler_worm_up = \
        set_optimizer(model, training_configuration, train_loader, 
                      amount_of_class = amount_of_class, alpha = alpha)
    
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
    
    
    # run training 
    gc.collect()
    if training_configuration.train_model:
        train_results_df = main(model, student, optimizer, criterion, ranking_criterion, accuracy_metric , perm_creterion,
                                train_loader, val_loader, num_epochs=training_configuration.epochs_count, device=device,
                                tb_writer=tb_writer, max_opt = training_configuration.max_opt, 
                                model_path = model_path, scheduler = scheduler, scheduler_worm_up=scheduler_worm_up)
    # evaluate model 
    summary_modelresult_df = get_model_results(model, student, model_path, criterion,
                                  ranking_criterion, accuracy_metric, perm_creterion,
                                  train_loader,val_loader,test_loader, device,
                                  train_val_test_summary)


