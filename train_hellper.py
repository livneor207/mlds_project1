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

# def update_moving_average(ema_updater, student_model, teacher_model):
#     max_update_size = list(student_model.parameters()).__len__()-1
#     for idx, (teacher_params, student_params) in enumerate(zip(teacher_model.parameters(), student_model.parameters())):
#         # print(idx)
#         # get current weights
#         old_weight, up_weight = student_params.data, teacher_params.data
        
#         # update student weights
#         student_params.data = ema_updater.update_average(old_weight, up_weight)
       
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
    balance_factor: float = 1
    amount_of_patch: float = 25
    moving_average_decay: float = 0.01
    weight_decay: float = 1e-3
    def get_device_type(self):
        # check for GPU\CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
    def update_merics(self, loss_functions_name = 'CE', learning_rate = 1e-3, 
                      learning_type = 'supervised', batch_size = 8,
                      scheduler_name = 'OneCycleLR', max_opt = True,
                      epochs_count = 20, perm = 'no_perm', num_workers = 0,
                      max_lr = 1e-2, hidden_size = 512, balance_factor  = 1,
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
        self.balance_factor = balance_factor
        self.amount_of_patch = amount_of_patch
        self.moving_average_decay = moving_average_decay
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name

        
        
def set_rank_metrics(metric_name = 'KendallRankCorrCoef', num_outputs = 2):
    """
    example
    from torchmetrics.regression import KendallRankCorrCoef
    preds = torch.tensor([[17, 15], [15, 17]], requires_grad=True)
    target = torch.tensor([[0, 1], [1, 0]])


    kendall = KendallRankCorrCoef(num_outputs=2)
    kendall(preds, target)
    """
    if metric_name == 'KendallRankCorrCoef':
        from torchmetrics.regression import KendallRankCorrCoef
        rank_metric = KendallRankCorrCoef(num_outputs=num_outputs)
        # kendall(preds, target)
    elif metric_name == 'RetrievalMAP':
        from torchmetrics import RetrievalMAP
        # indexes = tensor([0, 0, 0, 1, 1, 1, 1])
        # preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        # target = tensor([False, False, True, False, True, False, True])
        rank_metric = RetrievalMAP()
        # rmap(preds, target, indexes=indexes)
    elif metric_name == 'RetrievalMRR':
        from torchmetrics import RetrievalMRR
        
        rank_metric = RetrievalMRR()
        # indexes = tensor([0, 0, 0, 1, 1, 1, 1])
        # preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        # target = tensor([False, False, True, False, True, False, True])
        # mrr(preds, target, indexes=indexes)
    else:
        assert False, "no acceptable metric choosen" 


    return rank_metric

def set_rank_loss(loss_name = 'HingeEmbeddingLoss', margin = 1, num_labels = 1, beta=1):
    """
    example
    target = torch.tensor([[2,1,3,4,5], [2,1,3,4,5], [2,1,3,4,5]],requires_grad = True, dtype = torch.float)
    pred =  torch.tensor([[700,500,3,10,5], [2,40,3,5,0], [1000,701,500000,2000,704]],requires_grad = True, dtype = torch.float)
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
    ranking_criterion = torch.nn.MarginRankingLoss()

    loss = ranking_criterion(target_normelized, pred_normelized, target)

    loss.backward()
    """
   
    if loss_name == 'MarginRankingLoss':
        ranking_criterion = torch.nn.MarginRankingLoss(margin=margin, size_average=True, reduce=True, reduction='mean')
    elif loss_name == 'HingeEmbeddingLoss':
        ranking_criterion = nn.HingeEmbeddingLoss()
    elif loss_name == 'KLDivLoss':
        ranking_criterion = nn.KLDivLoss(reduction="batchmean")
        
        # target = torch.tensor([[2,1,3,4,5], [2,1,3,4,5], [2,1,3,4,5]],requires_grad = True, dtype = torch.float)
        # pred =  torch.tensor([[2,1,5,3,4], [2,4,3,4,1], [2,1,3,4,5]],requires_grad = True, dtype = torch.float)
        # pred = F.log_softmax(pred, dim=1)
        # target = F.softmax(target, dim=1)
        # output = kl_loss(pred, target)
    elif loss_name == 'MSE':
         ranking_criterion = torch.nn.MSELoss()
    elif loss_name == 'L1Loss':
         ranking_criterion = torch.nn.L1Loss()
    elif loss_name == 'SmoothL1Loss':
          ranking_criterion = torch.nn.SmoothL1Loss(beta=beta)
    else:
        assert False, "no acceptable loss choosen" 
    return ranking_criterion


def calculate_rank_loss(pred, target):
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
        factor = 0.3  # reduce by factor 0.5
        patience = 2  # epochs
        threshold = 1e-2
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


def prepare_for_rank_cretertion(target, pred):
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
    # yy2 =target.detach().numpy()
    # print(f"number of zeros: {yy2.size - np.count_nonzero(yy2)}")

    return target, target_normelized, pred_normelized


def calculate_rank_loss(ranking_criterion, target, pred):
    if isinstance(ranking_criterion, torch.nn.MarginRankingLoss):
        target, target_normelized, pred_normelized = prepare_for_rank_cretertion(target, pred)

        ranking_loss = ranking_criterion(target_normelized, pred_normelized, target)
    elif isinstance(ranking_criterion, torch.nn.MSELoss):
            ranking_loss = ranking_criterion(pred, target)
    elif isinstance(ranking_criterion, torch.nn.L1Loss):
            ranking_loss = ranking_criterion(pred, target)
    elif isinstance(ranking_criterion, torch.nn.SmoothL1Loss):
            ranking_loss = ranking_criterion(pred, target)
    elif isinstance(ranking_criterion, torch.nn.KLDivLoss):
        
        
        # criterion = torch.nn.L1Loss(reduce =False)
        # criterion = torch.nn.MSELoss(reduce =False)
        # ranking_loss = criterion(pred, target)
        # ranking_loss = torch.multiply(ranking_loss, (target/target.max())**9)
        # ranking_loss  = (ranking_loss.sum()/target.shape[0])
        ranking_loss = ranking_criterion(pred,target)
        # (target.argsort(1)-pred.argsort(1)==0).sum()
        
        # temperature_target, dummy = target.max(1)
        # temperature_pred, dummy = pred.max(1)
        
        # pred_probs_ordered = nn.functional.softmax(torch.div(pred.T, 1).T, dim=1)
        # y_true_probs_ordered = nn.functional.softmax(torch.div(target.T, 1).T, dim=1)

        # # Compute the KL divergence between the true and predicted probabilities
        
        # ranking_loss = ranking_criterion(torch.log(pred_probs_ordered), y_true_probs_ordered)
        
        
        
        

        # pred_norm = LA.norm(pred, 2)
        # target_norm = LA.norm(target, 2)
        # ((pred.argsort(1)-target)==0).sum()/target.numel()
        
        

        
        # max_pred = pred.max()
        # min_pred = pred.min()
        # max_target = target.max()
        # # yy = (pred-min_pred)/(max_pred-min_pred)
        # pred_normelized = max_target*((pred-min_pred)/(max_pred-min_pred))
        
        
        # target_normelized = (target-target.mean())
        # pred_normelized = (pred_normelized-target.mean())



        #pred_normelized = (pred-pred.mean())

        
        # max_target = target_normelized.max()
        # min_target = target_normelized.min()
        
        

        
        # pred_normelized = 2*(pred_normelized-min_target)/(max_target-min_target)
        
        
        # target_normelized = target_normelized/target_norm
        # pred_normelized = pred_normelized/pred_norm
        
        
        # pred_log_softmax = F.log_softmax(pred, dim=1)
        # target_softmax= F.softmax(target, dim=1)
        
        
        # pred_log_softmax = F.log_softmax(pred_normelized, dim=1)
        # target_softmax = F.softmax(target_normelized, dim=1)
        
        
        # ranking_loss = ranking_criterion(pred_log_softmax, target_softmax)
    
    elif isinstance(ranking_criterion, torch.nn.HingeEmbeddingLoss) :
        # pred_argsort,target_argsort  = pred_argsort.to(torch.float), target_argsort.to(torch.float)
        # projection1_norm = LA.norm(pred_argsort, 2)
        # projection2_norm = LA.norm(target_argsort, 2)
        # projection1_normelized = pred_argsort/projection1_norm
        # projection2_normelized = target_argsort/projection2_norm
        ranking_loss = ranking_criterion(projection1_normelized, projection2_normelized)

    return ranking_loss
def step(model, student, data, labels, criterion, ranking_criterion,  accuracy_metric, perm_order = None, optimizer=None):
    
    if data.shape[1]<=3:
        learning_type = 'supervised'
        m = torch.nn.Softmax(dim=1)
        if optimizer is  None:
          with torch.no_grad():
            classification_pred = model(data)
        else:
            classification_pred = model(data)
        criterion_loss = criterion(m(classification_pred), labels.argmax(1))
        
        _, predicted = torch.max(classification_pred.data, 1) # for getting predictions class
        _, labels_target = torch.max(labels.data, 1) # for getting predictions class
        f1_score = accuracy_metric(predicted, labels_target)

        accuracy = (predicted == labels_target).sum().item()/labels.shape[0] # get accuracy val

        if optimizer is not None:
            criterion_loss.backward()
            debug_grad= False
            if debug_grad:
                print_grad(model)
            optimizer.step()
        del classification_pred, data, labels_target

    else:
        learning_type = 'self_supervised'
        data2 = data[:,3::,:,:]
        data1 = data[:,0:3,:,:]
        prems_size  =  perm_order.shape[1]
        
        target_prem1 = perm_order[:,0,:]
        target_prem2 = perm_order[:,1,:]

        
        
        # target_prem1 = perm_order[:,0:prems_size//2]
        # target_prem2 = perm_order[:,prems_size//2::]
        
        # if optimizer is  None:
        #   with torch.no_grad():
        #     representation_pred_1_1 = model(data1)
        #     representation_pred_1_2 = model(data2)
        #     representation_pred_2_1 = student(data1)
        #     representation_pred_2_2 = student(data2)
        # else:
        #     representation_pred_1_1 = model(data1)
        #     torch.cuda.empty_cache()

        #     representation_pred_1_2 = model(data2)
        #     torch.cuda.empty_cache()

        #     representation_pred_2_1 = student(data1)
        #     torch.cuda.empty_cache()

        #     representation_pred_2_2 = student(data2)
        #     torch.cuda.empty_cache()

            
            
        
        
        
        if optimizer is  None:
          with torch.no_grad():
            representation_pred_1_1, perm_pred_1_1 = model(data1)
            torch.cuda.empty_cache()
            representation_pred_2_1, perm_pred_2_1 = student(data1)
            torch.cuda.empty_cache()
            del data1
            representation_pred_2_2, perm_pred_2_2 = student(data2)
            torch.cuda.empty_cache()
            representation_pred_1_2, perm_pred_1_2 = model(data2)
            torch.cuda.empty_cache()
            del data2
            
        else:
            representation_pred_1_1, perm_pred_1_1 = model(data1)
            torch.cuda.empty_cache()
            representation_pred_2_1, perm_pred_2_1 = student(data1)
            torch.cuda.empty_cache()
            del data1
            
            representation_pred_1_2, perm_pred_1_2 = model(data2)
            torch.cuda.empty_cache()
            
            representation_pred_2_2, perm_pred_2_2 = student(data2)
            torch.cuda.empty_cache()
            del data2
        del data
        """
        # representation_pred_1_1 = representation_pred_1_1.cpu()
        # representation_pred_1_2 = representation_pred_1_2.cpu()
        # representation_pred_2_1 = representation_pred_2_1.cpu()
        # representation_pred_2_2 = representation_pred_2_2.cpu()
        
        # perm_pred_1_1 = perm_pred_1_1.cpu()
        # perm_pred_1_2 = perm_pred_1_2.cpu()
        # perm_pred_2_1 = perm_pred_2_1.cpu()
        # perm_pred_2_2 = perm_pred_2_2.cpu()
        
        # projection1_norm = LA.norm(projection1, 2)
        # projection2_norm = LA.norm(projection2, 2)
        # projection1_normelized = projection1/projection1_norm
        # projection2_normelized = projection2/projection2_norm
        """
        
        
    

        # ranking_loss_2_1 = calculate_rank_loss(ranking_criterion, target_prem1, perm_pred_2_1)
        ranking_loss_1_1 = calculate_rank_loss(ranking_criterion, target_prem1, perm_pred_1_1)
        # ranking_loss_2_2 = calculate_rank_loss(ranking_criterion, target_prem2, perm_pred_2_2)
        ranking_loss_1_2 = calculate_rank_loss(ranking_criterion, target_prem2, perm_pred_1_2)
        
        
        order_ratio = (target_prem1.argsort(1)-perm_pred_1_1.argsort(1)==0).sum()/(target_prem1.numel())+\
                        (target_prem2.argsort(1)-perm_pred_1_2.argsort(1)==0).sum()/(target_prem2.numel())
        # print(f'order ratio {order_ratio.item()}')

        del target_prem2,target_prem1,perm_pred_1_2,perm_pred_1_1

        
        rank_loss = ranking_loss_1_1 + ranking_loss_1_2
        balance_factor = model.balance_factor
        
        
        
        # representation_pred_1_1_norm = LA.norm(representation_pred_1_1, 2, dim =0)
        # representation_pred_2_2_norm = LA.norm(representation_pred_2_2, 2, dim =0)
        # representation_pred_1_2_norm = LA.norm(representation_pred_1_2, 2, dim =0)
        # representation_pred_2_1_norm = LA.norm(representation_pred_2_1, 2, dim =0)
        
        # representation_pred_1_1_normelized = torch.div(representation_pred_1_1,representation_pred_1_1_norm)
        # representation_pred_2_2_normelized = torch.div(representation_pred_2_2,representation_pred_2_2_norm)
        # representation_pred_1_2_normelized = torch.div(representation_pred_1_2,representation_pred_1_2_norm)
        # representation_pred_2_1_normelized = torch.div(representation_pred_2_1,representation_pred_2_1_norm)

        
        # criterion(torch.Tensor([[-0.5,0.5]]), torch.Tensor([[1,0.5]]))
        
        # similiarities_loss = criterion(representation_pred_1_1_normelized, representation_pred_2_2_normelized)
        similiarities_loss = criterion(representation_pred_1_1, representation_pred_2_2)

        # criterion_loss1 = 2-2*similiarities_loss
        criterion_loss1 = similiarities_loss

        # criterion_loss1 = criterion_loss1.mean()
        
       
        # similiarities_loss = criterion(representation_pred_1_2_normelized, representation_pred_2_1_normelized)
        similiarities_loss = criterion(representation_pred_1_2, representation_pred_2_1)

        # criterion_loss2 = 2-2*similiarities_loss
        criterion_loss2 = similiarities_loss

        # criterion_loss2 = criterion_loss2.mean()

        criterion_loss = criterion_loss1 + criterion_loss2
        criterion_loss = criterion_loss.mean()
        if balance_factor != 0:
            criterion_loss *= balance_factor 
        else:
            rank_loss *= balance_factor
        accuracy = criterion_loss.item()
        f1_score = rank_loss
        # f1_score = torch.Tensor([0])
        
        criterion_loss += rank_loss
        if optimizer is not None:
            criterion_loss.backward()
            debug_grad= False
            if debug_grad:
                print_grad(model)
            
                     
            optimizer.step()
            
        _, predicted = None, None # for getting predictions class
        _, labels_target = None, None
        
        # accuracy = 0
        del representation_pred_1_2,representation_pred_2_1,representation_pred_1_1,representation_pred_2_2

        
    
                
    return criterion_loss, accuracy, f1_score


def eval_model(model, student, classification_criterion, ranking_criterion, accuracy_metric, data_loader, device):
    total_accuracy = 0.
    total_f1_score = 0.
    total_classification_loss = 0.
    debug = False
    model.eval()
    for idx, (data, target, perm_order, target_name) in enumerate(data_loader):
        batch_size = target.shape[0]
        torch.cuda.empty_cache()
        gc.collect()
        if idx>1 and debug:
            break
        classification_loss, accuracy, f1_score =  \
            step(model, student,  data.to(device), target.to(device), classification_criterion.to(device), ranking_criterion.to(device), accuracy_metric.to(device), perm_order.to(device) )
        del data, target, perm_order , target_name
        gc.collect()
        total_accuracy += accuracy*batch_size
        total_f1_score += f1_score*batch_size
        total_classification_loss += classification_loss*batch_size
    gc.collect()    
    total_accuracy =  np.round(total_accuracy/ data_loader.dataset.__len__(), 3)
    total_f1_score =  np.round(total_f1_score.item() /data_loader.dataset.__len__(), 3)
    total_classification_loss =  np.round(total_classification_loss.item() / data_loader.dataset.__len__(),3)

    return total_accuracy, total_f1_score, total_classification_loss


def train(model, student, optimizer, classification_criterion,
          ranking_criterion, accuracy_metric, data_loader, 
          device, scheduler= None, epoch= 1, num_epochs=1):
    total_accuracy = 0.
    total_f1_score = 0
    total_classification_loss = 0
    debug = False
    message = 'accuracy {}, f1-score {}, classification loss {}'
    with tqdm(data_loader) as pbar:
        model.train()
        for idx, (data, target, perm_order , target_name)  in enumerate(pbar):
            batch_size = target.shape[0]

            

            if idx >1 and debug:
                break
            optimizer.zero_grad()
            classification_loss, accuracy, f1_score  = step(model,student, data.to(device), target.to(device), classification_criterion.to(device), ranking_criterion.to(device), accuracy_metric.to(device), perm_order.to(device), optimizer)
            del data, target, perm_order , target_name
            gc.collect()
            if not student is  None:
                beta = model.student_ema_updater.initial_beta
                epoch_optimization_steps = data_loader.dataset.__len__()//batch_size
                total_amount_of_steps =  epoch_optimization_steps*num_epochs*2
                current_steps = epoch*batch_size+idx
                new_beta =  1-(1-beta)*(np.cos(((np.pi*current_steps)/(total_amount_of_steps)))+1)/2
                model.student_ema_updater.beta = new_beta
                update_moving_average(model.student_ema_updater, student, model)

            
            total_accuracy += accuracy*batch_size
            total_f1_score += f1_score*batch_size
            total_classification_loss += classification_loss*batch_size
            pbar.set_description(message.format(np.round(accuracy,3), \
                                                np.round(f1_score.item(),3), np.round(classification_loss.item(),3)))
            pbar.update()
           
    gc.collect()     
    total_accuracy =  np.round(total_accuracy /  data_loader.dataset.__len__(),3)
    total_f1_score =  np.round(total_f1_score.item() /  data_loader.dataset.__len__(),3)
    total_classification_loss =  np.round(total_classification_loss.item() / data_loader.dataset.__len__(),3)

    return total_accuracy, total_f1_score, total_classification_loss
            
def main(model, student, optimizer, classification_criterion, ranking_criterion, accuracy_metric, 
         train_loader, val_loader, num_epochs, device, tb_writer = None, 
         scheduler = None, model_path = '', max_opt = True):
    accuracy_train_list = []
    accuracy_validation_list = []
    loss_train_list = []
    loss_validation_list = []
    
    results_list =  [] 
    columns_list =  [ 'train_accuracy', 'train_f_score', 'train_classification_loss',
                     'val_accuracy', 'val_f_score_loss', 'val_classification_loss'] 
    if max_opt:
        best_model_score  = 0
    else:
        best_model_score = 1e5
    max_patience = 9
    patience = 0
    for epoch in range(num_epochs):
        train_accuracy, train_f1_score, train_classification_loss = \
            train(model, student, optimizer, classification_criterion, 
                  ranking_criterion, accuracy_metric, train_loader, device,
                  scheduler=scheduler, epoch = epoch, num_epochs=num_epochs )
        val_accuracy, val_f1_score, val_classification_loss = eval_model(model, student, classification_criterion, ranking_criterion, accuracy_metric, val_loader, device)
        
        
        if max_opt:
            current_val = val_f1_score
        else:
            current_val = val_classification_loss
        if not scheduler is None:
           if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_classification_loss)
           else:
               scheduler.step()  
            
        # print current results
        print(f'Epoch Summary {epoch}:\n'+\
              f'1) Train: f1-score {train_f1_score}, '+ \
                  f'classification_loss {train_classification_loss}, Accuracy {train_accuracy}\n' + \
              f'2) Validation: f1-score loss {val_f1_score}, '+ \
                  f'classification_loss {val_classification_loss}, val acc {val_accuracy}')
        
       
        results  = [train_accuracy, train_f1_score, train_classification_loss,
                    val_accuracy, val_f1_score, val_classification_loss]
        results_list.append(results)
        if (max_opt and current_val >= best_model_score) or (not max_opt and current_val <= best_model_score):
            best_model_wts = copy.deepcopy(model.state_dict())
            if model_path!= '':
                torch.save(best_model_wts, model_path)

            best_model_score = current_val
            patience = 0
        patience += 1
        if patience>max_patience:
          print(f'validation f1 scoredoes not improve for {max_patience} epoch, therefore optimization is stop due early stoping condition')
          break
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
            
            # adding model weights to tensorboard as histogram
            # add_model_weights_as_histogram(model, tb_writer, epoch)
        
       
            
    # load best model
    model.load_state_dict(best_model_wts)
    if not tb_writer is None and train_loader.dataset.learning_type == 'supervised':
        # add pr curves to tensor board
        add_pr_curves_to_tensorboard(model, val_loader, 
                                     device, 
                                     tb_writer, epoch, num_classes = train_loader.dataset.amount_of_class)
      
        add_wrong_prediction_to_tensorboard(model, val_loader, device, tb_writer, 
                                            1, tag='Wrong_Predections', max_images=50)

    train_results_df =  pd.DataFrame(results_list, columns = columns_list)
    train_results_df['ephoch_index'] = np.arange(train_results_df.shape[0])
    return train_results_df


