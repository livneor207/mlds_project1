import torch
import copy
import gc 
import numpy as np
from tqdm import tqdm
from tensorboard_hellper import *
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from torchmetrics import F1Score
from torch.nn import functional as F
from focal_loss.focal_loss import FocalLoss
from lion_pytorch import Lion
import torch.nn as nn
import pandas as pd 
from torch import linalg as LA
from model_builder import *
from data_set_loader import *

class TrainingConfiguration:
    '''
    Describes configuration of the training process
    '''
    batch_size: int = 128 
    epochs_count: int = 50  
    init_learning_rate: float = 0.1  # initial learning rate for lr scheduler
    log_interval: int = 5  
    test_interval: int = 1  
    data_root: str = "/kaggle/input/pytorch-opencv-course-classification/" 
    num_workers: int = 2  
    device: str = 'cuda'  
    classification_loss_name: str = 'ce' 
    learning_rate: float = 1e-3
    max_lr: float = 1e-2
    weight_decay: float = 1e-3
    scheduler_name: str = 'OneCycleLR'
    scheduler_name: str = 'None'
    optimizer_name: str = 'lion'
    learning_type: str  = 'supervised'
    max_opt : bool = True
    perm: str = 'no_perm'
    num_workers: int = 0
    hidden_size: int = 512
    amount_of_patch: float = 25
    moving_average_decay: float = 0.01
    weight_decay: float = 1e-3
    postion_embedding_balance_factor: float = 1
    permutation_prediction_balance_factor: float = 1

    def get_device_type(self):
        # check for GPU\CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
    def update_merics(self, loss_functions_name = 'CE', learning_rate = 1e-3, 
                      learning_type = 'supervised', batch_size = 8,
                      scheduler_name = 'OneCycleLR', max_opt = True,
                      epochs_count = 20, perm = 'no_perm', num_workers = 0,
                      max_lr = 1e-2, hidden_size = 512,
                      postion_embedding_balance_factor  = 1,
                      permutation_prediction_balance_factor = 1, 
                      amount_of_patch = 25, moving_average_decay = 0.01,
                      weight_decay = 1e-3, optimizer_name = 'lion'):
        
        self.loss_functions_name = loss_functions_name
        self.learning_rate = learning_rate
        self.learning_type = learning_type
        self.batch_size = batch_size
        self.scheduler_name = scheduler_name
        self.max_opt = max_opt
       
        self.epochs_count = epochs_count
        self.perm = perm
        self.num_workers = num_workers
        self.max_lr = max_lr
        self.hidden_size = hidden_size
        self.amount_of_patch = amount_of_patch
        self.moving_average_decay = moving_average_decay
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.permutation_prediction_balance_factor = permutation_prediction_balance_factor
        self.postion_embedding_balance_factor = postion_embedding_balance_factor

        
        
def set_postion_embedding_metrics(metric_name = 'KendallRankCorrCoef', num_outputs = 2):
    """
    examples
        # indexes = tensor([0, 0, 0, 1, 1, 1, 1])
        # preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        # target = tensor([False, False, True, False, True, False, True])
    """
    if metric_name == 'KendallRankCorrCoef':
        from torchmetrics.regression import KendallRankCorrCoef
        rank_metric = KendallRankCorrCoef(num_outputs=num_outputs)
    elif metric_name == 'RetrievalMAP':
        from torchmetrics import RetrievalMAP
        rank_metric = RetrievalMAP()
    elif metric_name == 'RetrievalMRR':
        from torchmetrics import RetrievalMRR
        rank_metric = RetrievalMRR()
    else:
        assert False, "no acceptable metric chosen"

    return rank_metric

def set_postion_embedding_loss(loss_name = 'HingeEmbeddingLoss', margin = 1, num_labels = 1, beta=1):
    if loss_name == 'MarginRankingLoss':
        postion_embedding_criterion = torch.nn.MarginRankingLoss(margin=margin, size_average=True, reduce=True, reduction='mean')
    elif loss_name == 'HingeEmbeddingLoss':
        postion_embedding_criterion = nn.HingeEmbeddingLoss()
    elif loss_name == 'KLDivLoss':
        postion_embedding_criterion = nn.KLDivLoss(reduction="batchmean")
    elif loss_name == 'MSE':
         postion_embedding_criterion = torch.nn.MSELoss()
    elif loss_name == 'L1Loss':
         postion_embedding_criterion = torch.nn.L1Loss()
    elif loss_name == 'SmoothL1Loss':
          postion_embedding_criterion = torch.nn.SmoothL1Loss(beta=beta)
    elif loss_name == 'CosineSimilarity':
        postion_embedding_criterion = nn.CosineSimilarity(dim=1, eps=1e-6)
    else:
        assert False, "no acceptable loss chosen"
    return postion_embedding_criterion


def calculate_postion_embedding_loss(pred, target):
    target_argsort = torch.argsort(target, dim=1)
    pred_argsort = torch.argsort(pred, dim=1)
    target_argsort = convert_2_float_and_require_grad(target_argsort)
    pred_argsort = convert_2_float_and_require_grad(pred_argsort)
    # target = (target_argsort-torch.arange(0,target.shape[1])).sign()
    index = (target_argsort-pred_argsort).sign()
    loss_val = loss(target_argsort, pred_argsort, index)
    return loss_val
def convert_2_float_and_require_grad(tensor):
    tensor = tensor.to(torch.float)
    return tensor

def set_similiarities_loss(classification_loss_name = 'CosineSimilarity', beta = 1):
    loss_name = classification_loss_name 
    
    loss_bank = ['CosineSimilarity', 'MSE', 'L1Loss', 'SmoothL1Loss']
    if not loss_name in loss_bank:
        assert False, 'loss is not defined'
    
    if loss_name == 'CosineSimilarity':
       criterion = nn.CosineSimilarity(dim=1, eps=1e-6)
    elif loss_name == 'MSE':
        criterion = torch.nn.MSELoss()
    elif loss_name == 'L1Loss':
         criterion = torch.nn.L1Loss()
    elif loss_name == 'SmoothL1Loss':
          criterion = torch.nn.SmoothL1Loss(beta=beta)
    return criterion

def set_classifcation_loss(training_configuration, alpha = None):
    loss_name = training_configuration.classification_loss_name 
    learning_rate = training_configuration.learning_rate
    device = training_configuration.device
    optimizer_name = training_configuration.optimizer_name
    
    loss_bank = ['fl', 'ce']
    if not loss_name in loss_bank:
        assert False, 'loss is not defined'
    
    if loss_name == 'fl':
        if alpha is None:
            criterion = FocalLoss(gamma=gamma)
        else:
            weights = torch.Tensor(1-alpha)
            gamma = 2
            criterion = FocalLoss(gamma=gamma, weights= weights)
    elif loss_name == 'ce':
          criterion = nn.CrossEntropyLoss()
  
    return criterion

def sellect_scheduler(optimizer, training_configuration, data_loader, scheduler_name = 'LambdaLR'):
    

    scheduler_bank = ['LambdaLR', 'OneCycleLR','ReduceLROnPlateau', 'None']
    if not scheduler_name in scheduler_bank:
        assert False, 'scheduler not defined'
    if scheduler_name == 'LambdaLR':
        # set decreasing schecdular
        decay_rate = 0.1
        lmbda = lambda epoch: 1/(1 + decay_rate * epoch)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
   
    elif scheduler_name == 'OneCycleLR':
        epochs_count = training_configuration.epochs_count
        max_lr = training_configuration.max_lr
        steps_per_epoch = len(data_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs_count)
    elif scheduler_name == 'ReduceLROnPlateau':
        factor = 0.5  # reduce by factor 0.5
        patience = 3  # epochs
        threshold = 1e-3
        verbose = True
        scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, verbose=verbose, threshold=threshold)
    elif scheduler_name == 'None':
        scheduler = None
    return scheduler


def set_optimizer(model, training_configuration, data_loader, amount_of_class = 13, alpha = None):
    loss_name = training_configuration.loss_functions_name 
    learning_rate = training_configuration.learning_rate
    device = training_configuration.device
    optimizer_name = training_configuration.optimizer_name
    scheduler_name = training_configuration.scheduler_name
    weight_decay = training_configuration.weight_decay
    
    optimizer_bank = ['adam', 'lion']
    if not optimizer_name in optimizer_bank:
       assert False, 'needed to add optimizer'
    # optimizer settings 
    if optimizer_name == 'adam':
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    elif optimizer_name == 'lion':
      optimizer = Lion(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    
    # Scheduler
    scheduler =  sellect_scheduler(optimizer, training_configuration, data_loader, scheduler_name = scheduler_name)
    
    return optimizer, scheduler

def set_metric(training_configuration, amount_of_class = 13, metric_name = 'accuracy'):
    loss_name = training_configuration.loss_functions_name 
    learning_rate = training_configuration.learning_rate
    device = training_configuration.device
    optimizer_name = training_configuration.optimizer_name
    metric_bank = ['f_score','accuracy']
    if not metric_name in metric_bank:
        assert False, 'metric is not defined '
    if metric_name == 'f_score':
        # f-score
        accuracy_metric = F1Score(task="multiclass", num_classes=amount_of_class, average =  'weighted')
    elif metric_name == 'accuracy':
        from torchmetrics.classification import MulticlassF1Score
        accuracy_metric =  MulticlassF1Score(num_classes=amount_of_class, average  = 'weighted')

    return accuracy_metric

def prepare_for_postion_embedding_critertion(target, pred):
    target_argsort = torch.argsort(target, dim=1)
    pred_argsort = torch.argsort(pred, dim=1)
    
    pred, target  = pred.to(torch.float), target.to(torch.float)
    pred_norm = LA.norm(pred, 2)
    target_norm = LA.norm(target, 2)
    
    target_normelized = (target-target.mean())/target_norm
    pred_normelized = (pred-pred.mean())/pred_norm

    target_normelized = convert_2_float_and_require_grad(target_normelized)
    pred_normelized = convert_2_float_and_require_grad(pred_normelized)
    target = (target_argsort-pred_argsort).sign()

    return target, target_normelized, pred_normelized


def calculate_postion_embedding_loss(postion_embedding_criterion, target, pred):
    if isinstance(postion_embedding_criterion, torch.nn.MarginRankingLoss):
        target, target_normelized, pred_normelized = prepare_for_postion_embedding_critertion(target, pred)
        postion_embedding_loss = postion_embedding_criterion(target_normelized, target_normalized, target)
    elif isinstance(postion_embedding_criterion, torch.nn.MSELoss):
        postion_embedding_loss = postion_embedding_criterion(pred, target)
    elif isinstance(postion_embedding_criterion, torch.nn.L1Loss):
        postion_embedding_loss = postion_embedding_criterion(pred, target)
    elif isinstance(postion_embedding_criterion, torch.nn.SmoothL1Loss):
        postion_embedding_loss = postion_embedding_criterion(pred, target)
    elif isinstance(postion_embedding_criterion, torch.nn.CosineSimilarity):
        postion_embedding_loss = torch.mean(2-2*postion_embedding_criterion(pred, target))
    elif isinstance(postion_embedding_criterion, torch.nn.KLDivLoss):
        postion_embedding_loss = postion_embedding_criterion(pred,target)
    elif isinstance(postion_embedding_criterion, torch.nn.HingeEmbeddingLoss):
        postion_embedding_loss = postion_embedding_criterion(pred,target)
    return postion_embedding_loss

def step(model, student, data, labels, criterion, postion_embedding_criterion,  
         accuracy_metric, permutation_classfication_creterion = None, 
         permutation_postion_embedding = None, permutation_label = None,  
         optimizer=None):
    
    # generate const for supervised mode 
    perm_classification_loss = torch.Tensor([0])
    f1_permutation_label_score = 0
    if model.learning_type == 'supervised':
        # classification step
        accuracy, criterion_loss, f1_score = \
            classification_step(accuracy_metric, criterion, data, labels, model, optimizer)
    else:
        # ssl step
        accuracy, criterion_loss, f1_permutation_label_score, f1_score, perm_classification_loss = \
            ssl_step(criterion, data,
                     f1_permutation_label_score,
                     model, optimizer,
                     perm_classification_loss,
                     permutation_classfication_creterion,
                     permutation_label,
                     permutation_postion_embedding,
                     postion_embedding_criterion,
                     student)

    return criterion_loss, accuracy, f1_score, f1_permutation_label_score, perm_classification_loss


def ssl_step(criterion, data, f1_permutation_label_score, model, optimizer, perm_classification_loss, permutation_classfication_creterion,
             permutation_label, permutation_postion_embedding, postion_embedding_criterion, student):
    # get loss parameters
    postion_embedding_balance_factor = model.postion_embedding_balance_factor
    permutation_prediction_balance_factor = model.permutation_prediction_balance_factor

    # parse datasets: images, posit    on embedding, permutation label
    data1, data2, target_prem1, target_prem2, \
        target_prem_label, target_prem_label2 = parse_ssl_data(data, permutation_label, permutation_postion_embedding)

    # forward sll model
    permutation_label_pred_1_1, permutation_label_pred_1_2, position_embedding_pred_1_1, \
        position_embedding_pred_1_2, representation_pred_1_1, representation_pred_1_2, \
        representation_pred_2_1, representation_pred_2_2 = forward_ssl_model(
        data1, data2, model, optimizer, student)
    
    # delete data (memory issues)
    del data
    
    # calculate position embedding loss
    postion_embedding_loss = ssl_postion_embedding_loss(position_embedding_pred_1_1, 
                                           position_embedding_pred_1_2, 
                                           postion_embedding_criterion,
                                           target_prem1, target_prem2)
    # calculate permutation loss
    perm_classification_loss = ssl_permutation_classification_loss(perm_classification_loss, permutation_classfication_creterion,
                                                                   permutation_label_pred_1_1, permutation_label_pred_1_2,
                                                                   target_prem_label, target_prem_label2)
    # calculate permutation order prediction
    f1_permutation_label_score = ssl_permutation_accuracy_prediction(f1_permutation_label_score, permutation_label_pred_1_1,
                                                              permutation_label_pred_1_2, target_prem_label,
                                                              target_prem_label2)
    # delete target data (memory issues)
    del target_prem2, target_prem1, position_embedding_pred_1_2, position_embedding_pred_1_1
    # calculate representation loss
    criterion_loss = representation_embedding_loss(criterion, representation_pred_1_1, representation_pred_1_2,
                                                   representation_pred_2_1, representation_pred_2_2)
    
    
    
    
    # calculate full loss
    criterion_loss, perm_classification_loss, postion_embedding_loss, accuracy, f1_score =  \
        calculate_complete_ssl_loss(postion_embedding_balance_factor,
        permutation_prediction_balance_factor,
        criterion_loss,
        perm_classification_loss,
        postion_embedding_loss)
    # delete input (memory issues)
    del representation_pred_1_2, representation_pred_2_1, representation_pred_1_1, representation_pred_2_2
    
    # optimization step
    backward_and_optimization_step(criterion_loss, model, optimizer)
    return accuracy, criterion_loss, f1_permutation_label_score, f1_score, perm_classification_loss


def backward_and_optimization_step(criterion_loss, model, optimizer):
    if optimizer is not None:
        criterion_loss.backward()
        debug_grad = False
        if debug_grad:
            print_grad(model)

        optimizer.step()


def calculate_complete_ssl_loss(postion_embedding_balance_factor,
                                permutation_prediction_balance_factor,
                                criterion_loss, perm_classification_loss, 
                                postion_embedding_loss):
    postion_embedding_loss *= postion_embedding_balance_factor
    perm_classification_loss *= permutation_prediction_balance_factor
    
    # update step for logg
    accuracy = criterion_loss.item()
    f1_score = postion_embedding_loss
    
    criterion_loss += postion_embedding_loss
    criterion_loss += perm_classification_loss
    return criterion_loss, perm_classification_loss, postion_embedding_loss, accuracy, f1_score


def representation_embedding_loss(criterion, representation_pred_1_1, representation_pred_1_2, representation_pred_2_1,
                                  representation_pred_2_2):
    if isinstance(criterion, torch.nn.CosineSimilarity):
        similiarities_loss1 = torch.mean(2 - 2 * criterion(representation_pred_1_1, representation_pred_2_2))
    else:
        similiarities_loss1 = criterion(representation_pred_1_1, representation_pred_2_2)
    if isinstance(criterion, torch.nn.CosineSimilarity):
        similiarities_loss2 = torch.mean(2 - 2 * criterion(representation_pred_1_2, representation_pred_2_1))
    else:
        similiarities_loss2 = criterion(representation_pred_1_2, representation_pred_2_1)
    criterion_loss = similiarities_loss1 + similiarities_loss2
    criterion_loss = criterion_loss.mean()
    return criterion_loss


def ssl_permutation_accuracy_prediction(f1_permutation_label_score, permutation_label_pred_1_1, permutation_label_pred_1_2,
                                        target_prem_label, target_prem_label2):
    _, permutation_label_pred_1_1 = torch.max(permutation_label_pred_1_1.data, 1)  # for getting predictions class
    _, permutation_label_pred_1_2 = torch.max(permutation_label_pred_1_2.data, 1)  # for getting predictions class
    _, target_prem_label = torch.max(target_prem_label.data, 1)  # for getting predictions class
    _, target_prem_label2 = torch.max(target_prem_label2.data, 1)  # for getting predictions class
    f1_permutation_label_score1 = (permutation_label_pred_1_1 == target_prem_label).sum().item() / target_prem_label.shape[
        0]  # get accuracy val
    f1_permutation_label_score2 = (permutation_label_pred_1_2 == target_prem_label2).sum().item() / target_prem_label.shape[
        0]  # get accuracy val
    f1_permutation_label_score = (f1_permutation_label_score1 + f1_permutation_label_score2) / 2
    return f1_permutation_label_score


def ssl_permutation_classification_loss(perm_classification_loss, permutation_classfication_creterion, permutation_label_pred_1_1,
                                        permutation_label_pred_1_2, target_prem_label, target_prem_label2):
    m = torch.nn.Softmax(dim=1)
    perm_classification_loss1 = permutation_classfication_creterion(m(permutation_label_pred_1_1), target_prem_label.argmax(1))
    perm_classification_loss2 = permutation_classfication_creterion(m(permutation_label_pred_1_2), target_prem_label2.argmax(1))
    perm_classification_loss = perm_classification_loss2 + perm_classification_loss1
    return perm_classification_loss


def ssl_postion_embedding_loss(position_embedding_pred_1_1, position_embedding_pred_1_2, postion_embedding_criterion, target_prem1, target_prem2):
    postion_embedding_loss_1_1 = calculate_postion_embedding_loss(postion_embedding_criterion, target_prem1, position_embedding_pred_1_1)
    postion_embedding_loss_1_2 = calculate_postion_embedding_loss(postion_embedding_criterion, target_prem2, position_embedding_pred_1_2)
    postion_embedding_loss = postion_embedding_loss_1_1 + postion_embedding_loss_1_2

    return postion_embedding_loss


def forward_ssl_model(data1, data2, model, optimizer, student):
    if optimizer is None:
        with torch.no_grad():
            representation_pred_1_1, position_embedding_pred_1_1, permutation_label_pred_1_1 = model(data1)
            torch.cuda.empty_cache()
            representation_pred_2_1, position_embedding_pred_2_1, dummy = student(data1)
            torch.cuda.empty_cache()
            del data1
            representation_pred_2_2, position_embedding_pred_2_2, dummy = student(data2)
            torch.cuda.empty_cache()
            representation_pred_1_2, position_embedding_pred_1_2, permutation_label_pred_1_2 = model(data2)
            torch.cuda.empty_cache()
            del data2

    else:
        representation_pred_1_1, position_embedding_pred_1_1, permutation_label_pred_1_1 = model(data1)
        torch.cuda.empty_cache()
        representation_pred_2_1, position_embedding_pred_2_1, dummy = student(data1)
        torch.cuda.empty_cache()
        del data1
        representation_pred_2_2, position_embedding_pred_2_2, dummy = student(data2)
        torch.cuda.empty_cache()
        representation_pred_1_2, position_embedding_pred_1_2, permutation_label_pred_1_2 = model(data2)
        torch.cuda.empty_cache()
        del data2
    
    return permutation_label_pred_1_1, permutation_label_pred_1_2, position_embedding_pred_1_1, position_embedding_pred_1_2, representation_pred_1_1, representation_pred_1_2, representation_pred_2_1, representation_pred_2_2


def parse_ssl_data(data, permutation_label, permutation_postion_embedding):
    data2 = data[:, 3::, :, :]
    data1 = data[:, 0:3, :, :]
    target_prem1 = permutation_postion_embedding[:, 0, :]
    target_prem2 = permutation_postion_embedding[:, 1, :]
    target_prem_label = permutation_label[:, 0, :]
    target_prem_label2 = permutation_label[:, 1, :]
    return data1, data2, target_prem1, target_prem2, target_prem_label, target_prem_label2


def classification_step(accuracy_metric, criterion, data, labels, model, optimizer):
    m = torch.nn.Softmax(dim=1)
    
    # forward
    classification_pred = supervised_forward(data, model, optimizer)
    
    # calculate classification loss 
    criterion_loss = criterion(m(classification_pred), labels.argmax(1))
    
    # calculate accuracy 
    accuracy, f1_score = calculate_classification_accuracy(accuracy_metric, classification_pred, labels)
    
    # backward step
    classification_backward_step(criterion_loss, model, optimizer)
    
    # delete input (memory)
    del classification_pred, data
    return accuracy, criterion_loss, f1_score


def classification_backward_step(criterion_loss, model, optimizer):
    if optimizer is not None:
        criterion_loss.backward()
        debug_grad = False
        if debug_grad:
            print_grad(model)
        optimizer.step()


def calculate_classification_accuracy(accuracy_metric, classification_pred, labels):
    _, predicted = torch.max(classification_pred.data, 1)  # for getting predictions class
    _, labels_target = torch.max(labels.data, 1)  # for getting predictions class
    f1_score = accuracy_metric(predicted, labels_target)
    accuracy = (predicted == labels_target).sum().item() / labels.shape[0]  # get accuracy val
    del labels_target, predicted
    return accuracy, f1_score


def supervised_forward(data, model, optimizer):
    if optimizer is None:
        with torch.no_grad():
            classification_pred = model(data)
    else:
        classification_pred = model(data)
    return classification_pred


def validation_step(model, student, classification_criterion, postion_embedding_criterion, accuracy_metric, permutation_classfication_creterion, data_loader, device):
    total_accuracy = 0.
    total_f1_score = 0.
    total_classification_loss = 0.
    total_f1_permutation_score = 0.
    total_permutation_classification_loss = 0.
    
    debug = False
    model.eval()
    for idx, (data, target, permutation_postion_embedding, target_name, permutation_label) in enumerate(data_loader):
        batch_size = target.shape[0]
        torch.cuda.empty_cache()
        gc.collect()
        if idx>1 and debug:
            break
        classification_loss, accuracy, f1_score, f1_permutation_label_score, perm_classification_loss =  \
            step(model, student,  data.to(device), target.to(device), classification_criterion.to(device),
                 postion_embedding_criterion.to(device), accuracy_metric.to(device), permutation_classfication_creterion.to(device), 
                 permutation_postion_embedding.to(device), permutation_label.to(device))
        del data, target, permutation_postion_embedding , target_name
        gc.collect()
        
        total_f1_permutation_score += f1_permutation_label_score*batch_size
        total_permutation_classification_loss += perm_classification_loss*batch_size
        total_accuracy += accuracy*batch_size
        total_f1_score += f1_score*batch_size
        total_classification_loss += classification_loss*batch_size
    gc.collect()    
    
    total_accuracy =  np.round(total_accuracy/ data_loader.dataset.__len__(), 3)
    total_f1_score =  np.round(total_f1_score.item() /data_loader.dataset.__len__(), 3)
    total_classification_loss =  np.round(total_classification_loss.item() / data_loader.dataset.__len__(),3)
    total_permutation_classification_loss =  np.round(total_permutation_classification_loss.item() / data_loader.dataset.__len__(),3)
    total_f1_permutation_score =  np.round(total_f1_permutation_score / data_loader.dataset.__len__(),3)

    return total_accuracy, total_f1_score, total_classification_loss, total_permutation_classification_loss, total_f1_permutation_score


def train_step(model, student, optimizer, classification_criterion,
          postion_embedding_criterion, accuracy_metric, permutation_classfication_creterion,  data_loader, 
          device, scheduler= None, epoch= 1, num_epochs=1):
    total_accuracy = 0.
    total_f1_score = 0
    total_classification_loss = 0
    total_f1_permutation_score = 0
    total_permutation_classification_loss = 0

    
    debug = False
    if model.learning_type == 'supervised':
        message = 'accuracy {}, f1-score {}, classification loss {}'
    else:
        message = 'embeedding loss {}, postion embedding loss {}, total loss {}, classification permutation loss {}, permutation prediction accuracy {}'

    with tqdm(data_loader) as pbar:
        model.train()
        for idx, (data, target, permutation_postion_embedding , target_name, permutation_label)  in enumerate(pbar):
            batch_size = target.shape[0]

            

            if idx >1 and debug:
                break
            optimizer.zero_grad()
            classification_loss, accuracy, f1_score, f1_permutation_label_score, perm_classification_loss \
                = step(model,student, data.to(device), target.to(device), 
                        classification_criterion.to(device), postion_embedding_criterion.to(device), 
                        accuracy_metric.to(device), permutation_classfication_creterion.to(device), 
                        permutation_postion_embedding.to(device), permutation_label.to(device),  optimizer)
            del data, target, permutation_postion_embedding , target_name
            gc.collect()
            if not student is  None:
                beta = model.student_ema_updater.initial_beta
                epoch_optimization_steps = data_loader.dataset.__len__()//batch_size
                total_amount_of_steps =  epoch_optimization_steps*num_epochs*2
                current_steps = epoch*batch_size+idx
                new_beta =  1-(1-beta)*(np.cos(((np.pi*current_steps)/(total_amount_of_steps)))+1)/2
                model.student_ema_updater.beta = new_beta
                update_moving_average(model.student_ema_updater, student, model)

            

            total_f1_permutation_score += f1_permutation_label_score*batch_size
            total_permutation_classification_loss += perm_classification_loss*batch_size
            total_accuracy += accuracy*batch_size
            total_f1_score += f1_score*batch_size
            total_classification_loss += classification_loss*batch_size
            pbar.set_description(message.format(np.round(accuracy,3), \
                                                np.round(f1_score.item(),3),\
                                                np.round(classification_loss.item(),3),
                                                np.round(perm_classification_loss.item(),3),
                                                np.round(f1_permutation_label_score,3)))
            pbar.update()
           
    gc.collect()     
    total_accuracy = np.round(total_accuracy / data_loader.dataset.__len__(),3)
    total_f1_score = np.round(total_f1_score.item() / data_loader.dataset.__len__(),3)
    total_classification_loss =  np.round(total_classification_loss.item() / data_loader.dataset.__len__(),3)
    total_permutation_classification_loss =  np.round(total_permutation_classification_loss.item() / data_loader.dataset.__len__(),3)
    total_f1_permutation_score =  np.round(total_f1_permutation_score / data_loader.dataset.__len__(),3)
    return total_accuracy, total_f1_score, total_classification_loss, total_permutation_classification_loss, total_f1_permutation_score
            
def training_loop(model, student, optimizer, classification_criterion, postion_embedding_criterion, accuracy_metric, permutation_classfication_creterion,
         train_loader, val_loader, num_epochs, device, tb_writer = None, 
         scheduler = None, model_path = '', max_opt = True):
    accuracy_train_list = []
    accuracy_validation_list = []
    loss_train_list = []
    loss_validation_list = []
    
    results_list =  []

    # generate columns for optimization summary
    columns_list = generate_summary_columns(model)

    # initialized best results
    best_model_score = initilizied_best_result(max_opt)

    # set early stopping parameters
    max_patience, patience = set_early_stoping_parameters()

    for epoch in range(num_epochs):

        # train step
        train_accuracy, train_f1_score, train_classification_loss, \
        train_permutation_classification_loss, train_f1_permutation_score = \
            train_step(model, student, optimizer, classification_criterion,
                  postion_embedding_criterion, accuracy_metric, permutation_classfication_creterion, train_loader, device,
                  scheduler=scheduler, epoch = epoch, num_epochs=num_epochs )
        
        
        # validation step
        val_accuracy, val_f1_score, \
        val_classification_loss, val_permutation_classification_loss, \
        val_f1_permutation_score = validation_step(model, student, classification_criterion,
                                         postion_embedding_criterion, accuracy_metric, permutation_classfication_creterion,
                                         val_loader, device)

        current_val = update_ephoch_result(max_opt, val_classification_loss, val_f1_score)

        # update schedular
        schedular_step(scheduler, val_classification_loss)

        # print epoch results
        print_epoch_results(epoch, model, train_accuracy, train_classification_loss, train_f1_permutation_score,
                            train_f1_score, train_permutation_classification_loss, val_accuracy,
                            val_classification_loss, val_f1_permutation_score, val_f1_score,
                            val_permutation_classification_loss)

        # add epoch result for summary
        add_apoch_results(model, results_list, train_accuracy, train_classification_loss, train_f1_permutation_score,
                          train_f1_score, train_permutation_classification_loss, val_accuracy, val_classification_loss,
                          val_f1_permutation_score, val_f1_score, val_permutation_classification_loss)

        # save model summary results
        train_results_df = save_training_summary_results(columns_list, model_path, results_list)

        # check if model was improve in this epoch
        best_model_wts, patience = optimization_improve_checker(best_model_score, current_val, max_opt, model,
                                                                model_path, patience)
        if patience>max_patience:
          declare_early_stopping_condition(max_patience, model)
          break

        # write scalars to tensorboard
        write_scalar_2_tensorboard(epoch, tb_writer, train_accuracy, train_classification_loss, train_f1_score,
                                    val_accuracy, val_classification_loss, val_f1_score)
            

    # load best model
    model.load_state_dict(best_model_wts)

    write_final_results_to_tensorboard(device, epoch, model, tb_writer, train_loader, val_loader)

    train_results_df = save_training_summary_results(columns_list, model_path, results_list)

    return train_results_df


def update_ephoch_result(max_opt, val_classification_loss, val_f1_score):
    if max_opt:
        current_val = val_f1_score
    else:
        current_val = val_classification_loss
    return current_val


def write_final_results_to_tensorboard(device, epoch, model, tb_writer, train_loader, val_loader):
    if not tb_writer is None and train_loader.dataset.learning_type == 'supervised':
        # add pr curves to tensor board
        add_pr_curves_to_tensorboard(model, val_loader,
                                     device,
                                     tb_writer, epoch, num_classes=train_loader.dataset.amount_of_class)

        add_wrong_prediction_to_tensorboard(model, val_loader, device, tb_writer,
                                            1, tag='Wrong_Predections', max_images=50)


def write_scalar_2_tensorboard(epoch, tb_writer, train_accuracy, train_classification_loss, train_f1_score,
                                val_accuracy, val_classification_loss, val_f1_score):
    if not tb_writer is None:
        # add scalar (loss/accuracy) to tensorboard
        tb_writer.add_scalar('Loss/Loss', val_classification_loss, epoch)
        tb_writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        tb_writer.add_scalar('F_score/Validation', val_f1_score, epoch)

        # add scalars (loss/accuracy) to tensorboard
        tb_writer.add_scalars('Loss/train-val', {'train': train_classification_loss,
                                                 'validation': val_classification_loss}, epoch)
        tb_writer.add_scalars('Accuracy/train-val', {'train': train_accuracy,
                                                     'validation': val_accuracy}, epoch)

        tb_writer.add_scalars('F_score/train-val', {'train': train_f1_score,
                                                    'validation': val_f1_score}, epoch)


def declare_early_stopping_condition(max_patience, model):
    if model.learning_type == 'supervised':
        print(
            f'validation f1 score does not improve for {max_patience} epoch, therefore optimization is stop due early stoping condition')
    else:
        print(
            f'validation total loss score does not improve for {max_patience} epoch, therefore optimization is stop due early stoping condition')


def optimization_improve_checker(best_model_score, current_val, max_opt, model, model_path, patience):
    if (max_opt and current_val >= best_model_score) or (not max_opt and current_val <= best_model_score):
        best_model_wts = copy.deepcopy(model.state_dict())
        if model_path != '':
            torch.save(best_model_wts, model_path)

        best_model_score = current_val
        patience = 0
    patience += 1
    return best_model_wts, patience


def save_training_summary_results(columns_list, model_path, results_list):
    train_results_df = pd.DataFrame(results_list, columns=columns_list)
    train_results_df['ephoch_index'] = np.arange(train_results_df.shape[0])
    csv_path = change_file_ending(model_path, '.csv')
    train_results_df.to_csv(csv_path)
    return train_results_df


def add_apoch_results(model, results_list, train_accuracy, train_classification_loss, train_f1_permutation_score,
                      train_f1_score, train_permutation_classification_loss, val_accuracy, val_classification_loss,
                      val_f1_permutation_score, val_f1_score, val_permutation_classification_loss):
    if model.learning_type == 'supervised':
        results = [train_accuracy, train_f1_score, train_classification_loss,
                   val_accuracy, val_f1_score, val_classification_loss]
    else:

        results = [train_accuracy, train_f1_score,
                   train_classification_loss, train_f1_permutation_score,
                   train_permutation_classification_loss, val_accuracy,
                   val_f1_score, val_classification_loss,
                   val_f1_permutation_score, val_permutation_classification_loss]
    results_list.append(results)


def set_early_stoping_parameters():
    max_patience = 9
    patience = 0
    return max_patience, patience


def initilizied_best_result(max_opt):
    if max_opt:
        best_model_score = 0
    else:
        best_model_score = 1e5
    return best_model_score


def generate_summary_columns(model):
    if model.learning_type == 'supervised':
        columns_list = ['train_accuracy', 'train_f_score', 'train_classification_loss',
                        'val_accuracy', 'val_f_score_loss', 'val_classification_loss']
    else:
        columns_list = ['train_embedding loss', 'train_position_embedding_loss',
                        'train_total_loss', 'train_f1_permutation_score',
                        'train_permutation_classification_loss', 'val_embedding loss',
                        'val_position_embedding_loss', 'val_total_loss',
                        'val_f1_permutation_score',
                        'val_permutation_classification_loss']
    return columns_list


def schedular_step(scheduler, val_classification_loss):
    if not scheduler is None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_classification_loss)
        else:
            scheduler.step()

            # print epoch results


def print_epoch_results(epoch, model, train_accuracy, train_classification_loss, train_f1_permutation_score,
                        train_f1_score, train_permutation_classification_loss, val_accuracy, val_classification_loss,
                        val_f1_permutation_score, val_f1_score, val_permutation_classification_loss):
    if model.learning_type == 'supervised':
        print(7 * '' + f'Epoch Summary {epoch}:\n' + \
              f'1) Train: f1-score {train_f1_score}, ' + \
              f'classification_loss {train_classification_loss}, ' + \
              f'Accuracy {train_accuracy}\n' + \
              f'2) Validation: f1-score {val_f1_score}, ' + \
              f'classification_loss {val_classification_loss}, ' + \
              f'val acc {val_accuracy}')



    else:
        print(7 * '' + f'Epoch Summary {epoch}:\n' + \
              f'1) Train: postion embedding loss {train_f1_score}, ' + \
              f'embeddding loss {train_accuracy}, ' + \
              f'total loss {train_classification_loss}, ' + \
              f'permutation prediction accuracy {train_f1_permutation_score},  ' + \
              f'permutation classification loss {train_permutation_classification_loss}\n'
              f'2) Validation: postion embedding loss {val_f1_score}, ' + \
              f'embedding loss {val_accuracy}, ' + \
              f'total loss {val_classification_loss}, ' + \
              f'permutation prediction accuracy {val_f1_permutation_score}, ' + \
              f'permutation classification loss {val_permutation_classification_loss}\n')


