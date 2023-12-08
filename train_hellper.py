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
# from flash.core.optimizers import LARS
from torch import linalg as LA
from model_builder import *
from data_set_loader import *
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
from torch.optim.optimizer import Optimizer, required

class LARS(Optimizer):
    def __init__(
    self,
    params,
    lr=required,
    momentum: float = 0,
    dampening: float = 0,
    weight_decay: float = 0,
    nesterov: bool = False,
    trust_coefficient: float = 0.001,
    eps: float = 1e-8,
    ):
            if lr is not required and lr < 0.0:
                raise ValueError(f"Invalid learning rate: {lr}")
            if momentum < 0.0:
                raise ValueError(f"Invalid momentum value: {momentum}")
            if weight_decay < 0.0:
                raise ValueError(f"Invalid weight_decay value: {weight_decay}")

            defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
            if nesterov and (momentum <= 0 or dampening != 0):
                raise ValueError("Nesterov momentum requires a momentum and zero dampening")

            self.eps = eps
            self.trust_coefficient = trust_coefficient

            super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # exclude scaling for params with 0 weight decay
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # lars scaling + weight decay part
                if weight_decay != 0:
                    if p_norm != 0 and g_norm != 0:
                        lars_lr = p_norm / (g_norm + p_norm * weight_decay + self.eps)
                        lars_lr *= self.trust_coefficient

                        d_p = d_p.add(p, alpha=weight_decay)
                        d_p *= lars_lr

                # sgd part
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group["lr"])

        return loss

def sellect_rows_contain_substring_in_col(df, substring, column):
    """  
    slice row from dataframe contain in specific columns the following substring
    """
    filtered_df = simulation_summary[simulation_summary[column].str.contains(substring, case=False)]
    print(filtered_df)
    return filtered_df


def collect_train_csv_summary(csv_folder_path, substring):
    """
    collect all csv file contain in specific folder the following substring
    """
    file_type_to_find =  os.path.join(csv_folder_path, '*'+ substring + '.csv')
    all_csv_path = glob.glob(file_type_to_find)
    if substring.find('ephocs')!=-1:
        results_columns = ['sim_name','train_accuracy', 'train_f_score', 'val_accuracy', 'val_f_score_loss']

    elif substring.find('summary')!=-1:
        results_columns = ['sim_name','train_accuracy', 'train_f1_score',
                            'val_accuracy', 'val_f1_score',
                            'test_accuracy', 'test_f1_score']
    else:
        print('substring to find is not defined')
        return
    if len(all_csv_path):
        result_collector = []
        for i_path in  all_csv_path:
            i_df =  pd.read_csv(i_path)
            sim_name = os.path.basename(i_path)
            i_df_columns = i_df.columns.tolist()
            if 'val_f1_permutation_score' in i_df_columns:
                continue
            else:
                if substring.find('ephocs')!=-1:
                    train_accuracy = i_df['train_accuracy'].max()
                    train_f_score = i_df['train_f_score'].max()
                    val_accuracy = i_df['val_accuracy'].max()
                    val_f_score = i_df['val_f_score_loss'].max()
                    i_results = [sim_name, train_accuracy, train_f_score, val_accuracy, val_f_score]
                else:
                    i_results = [sim_name] + i_df.values.tolist()[0]

                result_collector.append(i_results)
        simulation_summary = pd.DataFrame(result_collector, columns = results_columns)
        print(simulation_summary)
    else:
        print('no csv file has been find to find is not defined')
        
    return simulation_summary
        
        
        


def get_model_results(model, student, training_configuration, model_path, criterion,
                      ranking_criterion, accuracy_metric, perm_creterion,
                      train_loader, val_loader, test_loader, device,
                      train_val_test_summary):
    
    """ 
    load model best path, and run on all datasets in order to validate your model results 
    """
    
    # load model
    model = load_model(model_path, training_configuration)
    
    # send to device 
    model = model.to(device)

    # run on test 
    test_accuracy, test_f1_score, \
    test_classification_loss, test_perm_classification_loss, \
    test_f1_perm_score = eval_model(model, student, criterion,
                                     ranking_criterion, accuracy_metric, perm_creterion,
                                     test_loader, device)


    # run on train
    train_accuracy, train_f1_score, \
    train_classification_loss, train_perm_classification_loss, \
    train_f1_perm_score = eval_model(model, student, criterion,
                                     ranking_criterion, accuracy_metric, perm_creterion,
                                     train_loader, device)

    # run on val
    val_accuracy, val_f1_score, \
    val_classification_loss, val_perm_classification_loss, \
    val_f1_perm_score = eval_model(model, student, criterion,
                                     ranking_criterion, accuracy_metric, perm_creterion,
                                     val_loader, device)
    

    # collect model results into dataframe and write to disk
    columns_list = ['train_accuracy', 'train_f1_score',
                    'val_accuracy', 'val_f1_score',
                    'test_accuracy', 'test_f1_score']


    resluts_list = [[train_accuracy, train_f1_score,
                    val_accuracy, val_f1_score,
                    test_accuracy, test_f1_score]]

    result_df = pd.DataFrame(resluts_list, columns= columns_list)
    result_df.to_csv(train_val_test_summary)
    print(result_df)
    return result_df



def argparser_validation(argparser):
    """ 
    validate that the input to argparser are validate
    """
    max_allowed_permutation = argparser.max_allowed_permutation
    amount_of_patch = argparser.amount_of_patch
    perm = argparser.perm 
    
    # get amount of permutation 
    argparser.amount_of_perm = math.factorial(argparser.amount_of_patch) 
    
    # require to choose subgroup of permutation 
    if argparser.max_allowed_permutation <= argparser.amount_of_perm:
        argparser.max_allowed_permutation = max_allowed_permutation
    # amount of permutation is smaller than max allowed perm
    else:
        argparser.max_allowed_permutation =  argparser.amount_of_perm
    # if dont use permutation no need addition head factor 
    if perm != 'perm':
        argparser.balance_factor2 = argparser.balance_factor = 0
    
    # generate once permutations data
    all_permutation_option = generate_max_hamming_permutations(amount_of_perm = amount_of_patch, max_allowed_perm = max_allowed_permutation, amount_of_perm_to_generate = 50000)
    argparser.all_permutation_option = all_permutation_option
    
    # get available device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device = "mps:0"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    argparser.device = device
    
    # get grid size 
    argparser.grid_size = np.sqrt(argparser.amount_of_patch)
    mode_image_gridsize = argparser.image_dim%argparser.grid_size
    
    # fix case where defult image size is in dividable to grid size --> require correction to image size in order that image size will be dividable repect to grid size 
    if mode_image_gridsize != 0:
        addition_2_dim = int(argparser.grid_size-mode_image_gridsize)
        argparser.image_dim +=  addition_2_dim
        
    #  worm up condition
    if argparser.worm_up:
        if argparser.max_opt:
            argparser.worm_up = 0.75
        else:
            argparser.worm_up = 0.25
    # is worm up
    else:
        if argparser.max_opt:
            argparser.worm_up = 0
        else:
            argparser.worm_up = 1000000000000000000000

    

def update_merics(training_configuration, loss_functions_name = 'CE', learning_rate = 1e-3, 
                  learning_type = 'supervised', batch_size = 8,
                  scheduler_name = 'OneCycleLR', max_opt = True,
                  epochs_count = 20, perm = 'no_perm', num_workers = 0,
                  max_lr = 1e-2, hidden_size = 512, balance_factor  = 1,
                  balance_factor2 = 1, amount_of_patch = 9, moving_average_decay = 0.996,
                  weight_decay = 1e-5, optimizer_name = 'adam', max_allowed_permutation = 75,
                  use_auto_weight = False):
    
    """ 
    update 
    """
    training_configuration.loss_functions_name = loss_functions_name
    training_configuration.learning_rate = learning_rate
    training_configuration.learning_type = learning_type
    training_configuration.batch_size = batch_size
    training_configuration.scheduler_name = scheduler_name
    training_configuration.max_opt = max_opt
   
    training_configuration.epochs_count = epochs_count
    training_configuration.perm = perm
    training_configuration.num_workers = num_workers
    training_configuration.max_lr = max_lr
    training_configuration.hidden_size = hidden_size
    training_configuration.balance_factor = balance_factor
    training_configuration.amount_of_patch = amount_of_patch
    training_configuration.moving_average_decay = moving_average_decay
    training_configuration.weight_decay = weight_decay
    training_configuration.optimizer_name = optimizer_name
    training_configuration.balance_factor2 = balance_factor2
    training_configuration.amount_of_perm = math.factorial(training_configuration.amount_of_patch) 
    training_configuration.use_auto_weight = use_auto_weight
    
    # require to choosesubgroup of permutation 
    if max_allowed_permutation <= training_configuration.amount_of_perm:
        training_configuration.max_allowed_permutation = max_allowed_permutation
    # amount of permutation is smaller than max allowed perm
    else:
        training_configuration.max_allowed_permutation =  training_configuration.amount_of_perm
    # if dont use permutation no need addition head factor 
    if perm != 'perm':
        training_configuration.balance_factor2 = training_configuration.balance_factor = 0
 
    
    
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
    balance_factor: float = 1
    balance_factor2: float = 1
    max_allowed_permutation: int = 100
    use_auto_weight = False
    def get_device_type(self):
        # check for GPU\CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        self.device = device
    def update_merics(self, loss_functions_name = 'CE', learning_rate = 1e-3, 
                      learning_type = 'supervised', batch_size = 8,
                      scheduler_name = 'OneCycleLR', max_opt = True,
                      epochs_count = 20, perm = 'no_perm', num_workers = 0,
                      max_lr = 1e-2, hidden_size = 512, balance_factor  = 1,
                      balance_factor2 = 1, amount_of_patch = 9, moving_average_decay = 0.996,
                      weight_decay = 1e-5, optimizer_name = 'adam', max_allowed_permutation = 75,
                      use_auto_weight = False):
        
        
        """
        
        
        """
        # update parameters 
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
        self.balance_factor2 = balance_factor2
        self.amount_of_perm = math.factorial(self.amount_of_patch) 
        self.use_auto_weight = use_auto_weight
        
        # check i amount of permutation is greater than allowed  otherwise choose subgroup
        if max_allowed_permutation <= self.amount_of_perm:
            self.max_allowed_permutation = max_allowed_permutation
        # use all permutation 
        else:
            self.max_allowed_permutation =  self.amount_of_perm
        # for non permutation aug --> do not use addition head factors
        if perm != 'perm':
            self.balance_factor2 = self.balance_factor = 0
        
        # generate list of permutation 
        all_permutation_option = generate_max_hamming_permutations(amount_of_perm = amount_of_patch, max_allowed_perm = max_allowed_permutation, amount_of_perm_to_generate = 500000)
        self.all_permutation_option = all_permutation_option
        
        
def set_rank_metrics(metric_name = 'KendallRankCorrCoef', num_outputs = 2):
    """
    sey ranking metric (no grad)
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

def set_rank_loss(loss_name = 'CosineSimilarity', margin = 1, num_labels = 1, beta=1):
    """
    set postion embedding loss (with grad)
    mostly using cosin
    """
   
    if loss_name == 'MarginRankingLoss':
        ranking_criterion = torch.nn.MarginRankingLoss(margin=margin, size_average=True, reduce=True, reduction='mean')
    elif loss_name == 'HingeEmbeddingLoss':
        ranking_criterion = nn.HingeEmbeddingLoss()
    elif loss_name == 'KLDivLoss':
        ranking_criterion = nn.KLDivLoss(reduction="batchmean")
    elif loss_name == 'MSE':
         ranking_criterion = torch.nn.MSELoss()
    elif loss_name == 'L1Loss':
         ranking_criterion = torch.nn.L1Loss()
    elif loss_name == 'SmoothL1Loss':
          ranking_criterion = torch.nn.SmoothL1Loss(beta=beta)
    elif loss_name == 'CosineSimilarity':
        ranking_criterion = nn.CosineSimilarity(dim=1, eps=1e-6)
    else:
        assert False, "no acceptable loss choosen" 
    return ranking_criterion


def calculate_rank_loss(pred, target):
    """
    prepare vector to ranking loss    
    """
    target_argsort = torch.argsort(target, dim=1)
    pred_argsort = torch.argsort(pred, dim=1)
    target_argsort = convert_2_float_and_require_grad(target_argsort)
    pred_argsort = convert_2_float_and_require_grad(pred_argsort)
    index = (target_argsort-pred_argsort).sign()
    loss_val = loss(target_argsort, pred_argsort, index)
    return loss_val


def convert_2_float_and_require_grad(tensor):
    """ 
    convert vector into float in order to get grad 
    """
    tensor = tensor.to(torch.float)
    return tensor


def set_similiarities_loss(classification_loss_name = 'CosineSimilarity', beta = 1):
    """ 
    loss for measure how good representation we have 
    ussaly cosin
    """
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
    
    """ 
    set loss for classification usually cros entropy
    """
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
    """ 
    set scheduler for optimizer 
       
    worm-up is when we wish to start at very low learning rate in order to set good direction at the start of training,
    and as the trainig goes on the learning rate is increased 
    
    cooling up is the regular schedular
    """
    worm_up = training_configuration.worm_up
    learning_rate = training_configuration.learning_rate

    scheduler_bank = ['LambdaLR', 'OneCycleLR','ReduceLROnPlateau', 'None', 'worm_up']
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
    elif scheduler_name == 'worm_up' :  
        scheduler = None

    return scheduler


def set_optimizer(model, training_configuration, data_loader, amount_of_class = 13, alpha = None):
    """
    for supervised learning usallly using AdamW
    for ssl using Lars\AdamW (base paper )
    """
    loss_name = training_configuration.loss_functions_name 
    
    if model.learning_type == 'self_supervised':
        learning_rate = training_configuration.learning_rate_ssl
        optimizer_name = training_configuration.optimizer_name_ssl
        weight_decay = training_configuration.weight_decay_ssl

    else:
        learning_rate = training_configuration.learning_rate
        optimizer_name = training_configuration.optimizer_name
        weight_decay = training_configuration.weight_decay

    device = training_configuration.device
    scheduler_name = training_configuration.scheduler_name
    
    optimizer_bank = ['adam', 'lion', 'AdamW', 'Lars']
    include_params = [param for name, param in model.named_parameters() if 'sigma' not in name]


    if not optimizer_name in optimizer_bank:
       assert False, 'needed to add optimizer'
    # optimizer settings 
    if optimizer_name == 'adam':
      # model_parameters = [p for p in model.parameters()][1::] # remove sigma parameters
      optimizer = torch.optim.Adam(include_params, lr=learning_rate, weight_decay = weight_decay)
    elif optimizer_name == 'lion':
      optimizer = Lion(include_params, lr=learning_rate, weight_decay = weight_decay)
    elif optimizer_name == 'AdamW':
        # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = weight_decay, eps=1e-07)
        optimizer = torch.optim.AdamW(include_params, lr=learning_rate, weight_decay = weight_decay)
    elif optimizer_name == 'Lars':
        optimizer = LARS(include_params, lr=learning_rate, momentum=0.9, weight_decay= weight_decay)


    # Scheduler
    scheduler =  sellect_scheduler(optimizer, training_configuration, data_loader, scheduler_name = scheduler_name)
    
    # warm up scheduler 
    worm_up_scheduler =  sellect_scheduler(optimizer, training_configuration, data_loader, scheduler_name = 'worm_up')
    
    return optimizer, scheduler, worm_up_scheduler

def set_metric(training_configuration, amount_of_class = 13, metric_name = 'accuracy'):
    """ 
    set metric for accuracy\f-score from library
    """
    loss_name = training_configuration.loss_functions_name 
    learning_rate = training_configuration.learning_rate
    device = training_configuration.device
    optimizer_name = training_configuration.optimizer_name
    metric_bank = ['f_score','accuracy']
    if not metric_name in metric_bank:
        assert False, 'metric is not defined '
    # f-score
    if metric_name == 'f_score':   
        accuracy_metric = F1Score(task="multiclass", num_classes=amount_of_class, average =  'weighted')
    # accuracy 
    elif metric_name == 'accuracy':
        from torchmetrics.classification import MulticlassF1Score
        accuracy_metric =  MulticlassF1Score(num_classes=amount_of_class, average  = 'weighted')

    return accuracy_metric


def prepare_for_rank_cretertion(target, pred):
    """ 
    prepare to ranking loss
    """
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


def calculate_rank_loss(ranking_criterion, target, pred):
    """
    calculate postion embedding loss --> usually cosin 
    """
    if isinstance(ranking_criterion, torch.nn.MarginRankingLoss):
        target, target_normelized, pred_normelized = prepare_for_rank_cretertion(target, pred)

        ranking_loss = ranking_criterion(target_normelized, pred_normelized, target)
    elif isinstance(ranking_criterion, torch.nn.MSELoss):
            ranking_loss = ranking_criterion(pred, target)
    elif isinstance(ranking_criterion, torch.nn.L1Loss):
            ranking_loss = ranking_criterion(pred, target)
    elif isinstance(ranking_criterion, torch.nn.SmoothL1Loss):
            ranking_loss = ranking_criterion(pred, target)
    elif isinstance(ranking_criterion, torch.nn.CosineSimilarity):
            ranking_loss = torch.mean(2-2*ranking_criterion(pred, target))
    elif isinstance(ranking_criterion, torch.nn.KLDivLoss):
            ranking_loss = ranking_criterion(pred,target)
    elif isinstance(ranking_criterion, torch.nn.HingeEmbeddingLoss) :
        ranking_loss = ranking_criterion(projection1_normelized, projection2_normelized)
    return ranking_loss


def clip_gradient(model):
    """ 
    clip model graidint between 2 numbers 
    """
    # clip gradient between -1 to 1 
    for param in model.parameters():
        if param.grad is not None:
            max_norm = 0.1
            max_norm2 = collect_grads(model)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            param.grad.data.clamp_(-max_norm2, max_norm2)
            pass
        
        
        
def step(model, student, data, labels, criterion, ranking_criterion,  
         accuracy_metric, perm_creterion = None, 
         perm_order = None, perm_label = None,  
         optimizer=None, optimizer_sigma = None):
    
    """ 
    
    """
    # initlized postion embedding and perm label prediction scores
    perm_classification_loss = torch.Tensor([0])
    f1_perm_label_score = 0
    
    # is supervised task
    if data.shape[1]<=3:
        learning_type = 'supervised'
        # set softmax unit
        m = torch.nn.Softmax(dim=1)
        
        """
        forward step
        """
        # is not training step
        if optimizer is  None:
          # forward without gradient 
          with torch.no_grad():
            classification_pred = model(data)
        # training step
        else:
            classification_pred = model(data)
        
        # classification loss 
        criterion_loss = criterion(m(classification_pred), labels.argmax(1))
        
        # from probability prediction class 
        _, predicted = torch.max(classification_pred.data, 1) # for getting predictions class
        _, labels_target = torch.max(labels.data, 1) # for getting predictions class
        
        # calculate f-score 
        f1_score = accuracy_metric(predicted, labels_target)
        
        # calculate accuracy 
        accuracy = (predicted == labels_target).sum().item()/labels.shape[0] # get accuracy val

        # training phase 
        if optimizer is not None:
            
            # driviate 
            criterion_loss.backward()
            debug_grad= False
            if debug_grad:
                print_grad(model)
           
            # clip_gradient(model)
            
            # optimization step
            optimizer.step()
            
        # del variable in order 
        del classification_pred, data, labels_target
    # ssl 
    else:
        learning_type = 'self_supervised'
        
        # get image1\2
        data2 = data[:,3::,:,:]
        data1 = data[:,0:3,:,:]
        
        # get permutation index target  
        prems_size  =  perm_order.shape[1]
        target_prem_label = perm_label[:,0,:]
        target_prem_label2 = perm_label[:,1,:]
        
        # get position embedding target
        target_prem1 = perm_order[:,0,:]
        target_prem2 = perm_order[:,1,:]
        
        # del from memory issue         
        del perm_label, perm_order, data
        # check if is worming up
        is_worm_up = model.is_worm_up
        # is worm up --> start optimize addition head 
        if is_worm_up :
            balance_factor = model.balance_factor
            balance_factor2 = model.balance_factor2
        # is not worm up --> do not optimize addition head 
        else:
            balance_factor = balance_factor2 = 0
            
            
        # forward in non training phsase 
        if optimizer is  None:
             #  online model get image 1
             with torch.no_grad():
                representation_pred_1_1, perm_pred_1_1, perm_label_pred_1_1 = model(data1)
                torch.cuda.empty_cache()
             #  offline model get image 1
             with torch.no_grad():
                representation_pred_2_1, perm_pred_2_1, dummy = student(data1)
                torch.cuda.empty_cache()
             #  offline model get image 2
             with torch.no_grad():
                representation_pred_2_2, perm_pred_2_2, dummy = student(data2)
                torch.cuda.empty_cache()
             #  online model get image 2
             with torch.no_grad():
                representation_pred_1_2, perm_pred_1_2, perm_label_pred_1_2 = model(data2)
                torch.cuda.empty_cache()
        # forward in training phsase 
        else:
            #  online model get image 1
            representation_pred_1_1, perm_pred_1_1, perm_label_pred_1_1 = model(data1)
            torch.cuda.empty_cache()
            #  offline model get image 1 (no need grad)
            with torch.no_grad():
                representation_pred_2_1, perm_pred_2_1, dummy = student(data1)
            torch.cuda.empty_cache()
            #  offline model get image 2 (no need grad)
            with torch.no_grad():
                representation_pred_2_2, perm_pred_2_2, dummy = student(data2)
            torch.cuda.empty_cache()
            #  online model get image 2
            representation_pred_1_2, perm_pred_1_2, perm_label_pred_1_2 = model(data2)
            torch.cuda.empty_cache()
        
        # from memory issue 
        del data1, data2
        gc.collect()
        
        
        # get device 
        device = model.device
        
        # optmize postion embedding head 
        if balance_factor != 0:
            # calculate the ability of projected representation to predict the poation embedding 
            ranking_loss_1_1 = calculate_rank_loss(ranking_criterion, target_prem1, perm_pred_1_1)
            ranking_loss_1_2 = calculate_rank_loss(ranking_criterion, target_prem2, perm_pred_1_2)
            
            # sum the loss
            rank_loss = ranking_loss_1_1 + ranking_loss_1_2
            
        # no need to optimize postion embedding head 
        else:
            rank_loss = torch.Tensor([0])
            rank_loss = rank_loss.to(device)
        # optmize pemutation index prediction 
        if balance_factor2 != 0:
            m = torch.nn.Softmax(dim=1)
            # calculate cross entropy 
            perm_classification_loss1 = perm_creterion(m(perm_label_pred_1_1), target_prem_label.argmax(1))
            perm_classification_loss2 = perm_creterion(m(perm_label_pred_1_2), target_prem_label2.argmax(1))
        
            # summed the losses        
            perm_classification_loss = perm_classification_loss2 + perm_classification_loss1
            
            # from probability into prediction 
            _, perm_label_pred_1_1 = torch.max(perm_label_pred_1_1.data, 1) # for getting predictions class
            _, perm_label_pred_1_2 = torch.max(perm_label_pred_1_2.data, 1) # for getting predictions class
            _, target_prem_label = torch.max(target_prem_label.data, 1) # for getting predictions class
            _, target_prem_label2 = torch.max(target_prem_label2.data, 1) # for getting predictions class
        
            # calcualte accuracy in predcting premutation index  
            f1_perm_label_score1 = (perm_label_pred_1_1 == target_prem_label).sum().item()/target_prem_label.shape[0] # get accuracy val
            f1_perm_label_score2 = (perm_label_pred_1_2 == target_prem_label2).sum().item()/target_prem_label.shape[0] # get accuracy val
            
            # average the score for both images (1\2)
            f1_perm_label_score = (f1_perm_label_score1 + f1_perm_label_score2)/2

            del perm_label_pred_1_1, perm_label_pred_1_2, target_prem_label,target_prem_label2
            del target_prem2, target_prem1, perm_pred_1_2,perm_pred_1_1
        # no need to optimize permutation index prediction head
        else:
            f1_perm_label_score = 0
            perm_classification_loss = torch.Tensor([0])
            perm_classification_loss = perm_classification_loss.to(device)
         
        # calulate representation loss 
        # if loss is cosin
        if isinstance(criterion, torch.nn.CosineSimilarity):
            similiarities_loss =  torch.mean(2-2*criterion(representation_pred_1_1, representation_pred_2_2))
        # dist loss
        else:
            similiarities_loss = criterion(representation_pred_1_1, representation_pred_2_2)

        criterion_loss1 = similiarities_loss
        # if loss is cosin
        if isinstance(criterion, torch.nn.CosineSimilarity):
            similiarities_loss =  torch.mean(2-2*criterion(representation_pred_1_2, representation_pred_2_1))
        # dist loss
        else:
            similiarities_loss = criterion(representation_pred_1_2, representation_pred_2_1)

        criterion_loss2 = similiarities_loss

        # summed loss 
        criterion_loss = criterion_loss1 + criterion_loss2
        
        

        # get representation loss val
        accuracy = criterion_loss.item()
        # update postion embedding loss 
        f1_score = rank_loss
        # use auto weight optimization
        if model.use_auto_weight and is_worm_up:  

            sigma_squered = torch.pow(model.sigma,2)
            sigma1 = sigma_squered[0]
            sigma2 =  sigma_squered[1]
            sigma3 =  sigma_squered[2]
            # sigma1 = model.sigma[0]
            # sigma2 =  model.sigma[1]
            # sigma3 =  model.sigma[2]
            
            criterion_loss = criterion_loss/(sigma1*2) 
            rank_loss = rank_loss/(sigma2*2)
            perm_classification_loss = perm_classification_loss/(sigma3*2)
            
            constarint_sigma1 = torch.log(1+sigma1)
            constarint_sigma1 = constarint_sigma1.to(device)
            constarint_sigma2 = torch.log(1+sigma2)
            constarint_sigma2 = constarint_sigma2.to(device)
            constarint_sigma3 = torch.log(1+sigma3)
            constarint_sigma3 = constarint_sigma3.to(device)

            
            # constarint_sigma1 = torch.log(1+sigma_squered[0])
            # constarint_sigma1 = constarint_sigma1.to(device)
            # constarint_sigma2 = torch.log(1+sigma_squered[1])
            # constarint_sigma2 = constarint_sigma2.to(device)
            # constarint_sigma3 = torch.log(1+sigma_squered[2])
            # constarint_sigma3 = constarint_sigma3.to(device)
            if balance_factor !=0 :
                criterion_loss += constarint_sigma2
            if balance_factor2 !=0 :
                criterion_loss += constarint_sigma3  
            
            criterion_loss += constarint_sigma1
        else:
            # multi by head factors
            rank_loss *= balance_factor
            perm_classification_loss *= balance_factor2
            
            
        # optimize postion embedding head 
        if balance_factor != 0:    
            criterion_loss += rank_loss
            
        # optimize permutation indx prediction 
        if balance_factor2 != 0:    
            criterion_loss += perm_classification_loss
        
        # in training phase --> deriative             
        if optimizer is not None:
            criterion_loss.backward()
            debug_grad = False
            if debug_grad:
                print_grad(model)
            #clip_gradient(model)
            # optimize step
            optimizer.step()
            
            if not optimizer_sigma is None and model.use_auto_weight:
                optimizer_sigma.step()
            # validate that in auto task optimization weight are positive 
            model.sigma.data = torch.relu(model.sigma.data)
            # print(model.sigma)


        _, predicted = None, None # for getting predictions class
        _, labels_target = None, None
        
        del representation_pred_1_2, representation_pred_2_1
        del representation_pred_1_1, representation_pred_2_2


    return criterion_loss, accuracy, f1_score, f1_perm_label_score, perm_classification_loss


def eval_model(model, student, classification_criterion, ranking_criterion, accuracy_metric,perm_creterion, data_loader, device):
    """ 
    run model on all data
    """
    
    # initlized scores 
    total_accuracy = 0.
    total_f1_score = 0.
    total_classification_loss = 0.
    total_f1_perm_score = 0.
    total_perm_classification_loss = 0.
    
    debug = False
    
    # move into eval mode
    model.eval()
    
    # run on all data 
    for idx, (data, target, perm_order, target_name, perm_label) in enumerate(data_loader):
        batch_size = target.shape[0]
        if batch_size == 1:
            continue
        if idx>1 and debug:
            break
        
        # forward step 
        classification_loss, accuracy, f1_score, f1_perm_label_score, perm_classification_loss =  \
            step(model, student,  data.to(device), target.to(device), classification_criterion.to(device),
                 ranking_criterion.to(device), accuracy_metric.to(device), perm_creterion.to(device), 
                 perm_order.to(device), perm_label.to(device))
        del data, target, perm_order , target_name
        # gc.collect()
        
        # append results 
        total_f1_perm_score += f1_perm_label_score*batch_size
        total_perm_classification_loss += perm_classification_loss*batch_size
        total_accuracy += accuracy*batch_size
        total_f1_score += f1_score*batch_size
        total_classification_loss += classification_loss*batch_size
    # gc.collect()    
    
    total_accuracy =  np.round(total_accuracy/ data_loader.dataset.__len__(), 3)
    total_f1_score =  np.round(total_f1_score.item() /data_loader.dataset.__len__(), 3)
    total_classification_loss =  np.round(total_classification_loss.item() / data_loader.dataset.__len__(),3)
    total_perm_classification_loss =  np.round(total_perm_classification_loss.item() / data_loader.dataset.__len__(),3)
    total_f1_perm_score =  np.round(total_f1_perm_score / data_loader.dataset.__len__(),3)

    return total_accuracy, total_f1_score, total_classification_loss, total_perm_classification_loss, total_f1_perm_score


def train(model, student, optimizer, optimizer_sigma, classification_criterion,
          ranking_criterion, accuracy_metric, perm_creterion,  data_loader, 
          device, scheduler= None, epoch= 1, num_epochs=1):
    
    """ 
    training step
    
    
    
    """
    
    # intilized score 
    total_accuracy = 0.
    total_f1_score = 0
    total_classification_loss = 0
    total_f1_perm_score = 0
    total_perm_classification_loss = 0

    
    debug = False
    # set training log for supervised and for ssl 
    if model.learning_type == 'supervised':
        message = 'accuracy {}, f1-score {}, classification loss {}'
    else:
        message = 'embeedding loss {}, postion embedding loss {}, total loss {}, classification permutation loss {}, permutation prediction accuracy {}'
    all_idx = 0
    with tqdm(data_loader) as pbar:
        model.train()
        for idx, (data, target, perm_order , target_name, perm_label)  in enumerate(pbar):
            # get batch size 
            batch_size = target.shape[0]
            if batch_size == 1:
                continue
            if idx >1 and debug:
                break
            
            if not optimizer_sigma is None:
                optimizer_sigma.zero_grad()

            # initlized gradient 
            optimizer.zero_grad()
            
            # forward and optimize 
            classification_loss, accuracy, f1_score, f1_perm_label_score, perm_classification_loss \
                = step(model,student, data.to(device), target.to(device), 
                        classification_criterion.to(device), ranking_criterion.to(device), 
                        accuracy_metric.to(device), perm_creterion.to(device), 
                        perm_order.to(device), perm_label.to(device),  optimizer, optimizer_sigma)
            del data, target, perm_order , target_name
            # gc.collect()
            if not student is  None:
                # in ssl learning update beta (increase --> memory less)
                data_loader_batch_size = data_loader.batch_size
                epoch_optimization_steps = data_loader.dataset.__len__()//data_loader_batch_size+1
                current_steps = (epoch*epoch_optimization_steps+idx)
                beta = model.student_ema_updater.initial_beta
                total_amount_of_steps =  epoch_optimization_steps*(num_epochs*1)
                new_beta =  1-(1-beta)*(np.cos(((np.pi*current_steps)/(total_amount_of_steps)))+1)/2
                model.student_ema_updater.beta = new_beta
                if current_steps%50 == 0:
                    print('copy weights')
                    model.student_ema_updater.beta = 0
                    update_moving_average(model.student_ema_updater, student.backbone, model.backbone)
                    update_moving_average(model.student_ema_updater, student.REPRESENTATION_HEAD, model.REPRESENTATION_HEAD)
                    model.student_ema_updater.beta = new_beta
                else:

                    # update backbone and representation head
                    update_moving_average(model.student_ema_updater, student.backbone, model.backbone)
                    update_moving_average(model.student_ema_updater, student.REPRESENTATION_HEAD, model.REPRESENTATION_HEAD)

            # append results
            total_f1_perm_score += f1_perm_label_score*batch_size
            total_perm_classification_loss += perm_classification_loss*batch_size
            total_accuracy += accuracy*batch_size
            total_f1_score += f1_score*batch_size
            total_classification_loss += classification_loss*batch_size
            pbar.set_description(message.format(np.round(accuracy,3), \
                                                np.round(f1_score.item(),3),\
                                                np.round(classification_loss.item(),3),
                                                np.round(perm_classification_loss.item(),3),
                                                np.round(f1_perm_label_score,3)))
            pbar.update()
           
    # summary all batch results     
    total_accuracy =  np.round(total_accuracy /  data_loader.dataset.__len__(),3)
    total_f1_score =  np.round(total_f1_score.item() /  data_loader.dataset.__len__(),3)
    total_classification_loss =  np.round(total_classification_loss.item() / data_loader.dataset.__len__(),3)
    total_perm_classification_loss =  np.round(total_perm_classification_loss.item() / data_loader.dataset.__len__(),3)
    total_f1_perm_score =  np.round(total_f1_perm_score / data_loader.dataset.__len__(),3)
    return total_accuracy, total_f1_score, total_classification_loss, total_perm_classification_loss, total_f1_perm_score


def generate_summary_columns(model):
    """
    generate summary of training per ephoch for supervised and ssl 
    """
    if model.learning_type == 'supervised':
        columns_list = ['train_accuracy', 'train_f_score', 'train_classification_loss',
                        'val_accuracy', 'val_f_score_loss', 'val_classification_loss']
    else:
        columns_list = ['train_embedding_loss', 'train_position_embedding_loss','train_total_loss', 
                         'train_permutation_classification_loss','train_f1_permutation_score',
                         'val_embedding loss','val_position_embedding_loss', 'val_total_loss',
                         'val_f1_permutation_score', 'val_permutation_classification_loss']
        
            
    return columns_list

def set_early_stoping_parameters():
    """ 
    set early stoping codition 
    """
    max_patience = 9
    patience = 0
    return max_patience, patience

def initilizied_best_result(max_opt):
    """ 
    initlized best score for maximization and minimization task
    """
    if max_opt:
        best_model_score = 0
    else:
        best_model_score = 1e5
    return best_model_score

def print_epoch_results(epoch, model, train_accuracy, train_classification_loss, train_f1_permutation_score,
                        train_f1_score, train_permutation_classification_loss, val_accuracy, val_classification_loss,
                        val_f1_permutation_score, val_f1_score, val_permutation_classification_loss):
    
    """ 
    prnit log using model results 
    """
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
            
def add_apoch_results(model, results_list, train_accuracy, train_classification_loss, train_f1_permutation_score,
                      train_f1_score, train_permutation_classification_loss, val_accuracy, val_classification_loss,
                      val_f1_permutation_score, val_f1_score, val_permutation_classification_loss):
    
    """ 
    add apoch result for supervised and ssl 
    """
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
    return results_list

def declare_early_stopping_condition(max_patience, model):
    """ 
    decalere early stoping condition is achieved 
    """
    if model.learning_type == 'supervised':
        print(
            f'validation f1 score does not improve for {max_patience} epoch, therefore optimization is stop due early stoping condition')
    else:
        print(
            f'validation total loss score does not improve for {max_patience} epoch, therefore optimization is stop due early stoping condition')
def save_training_summary_results(columns_list, model_path, results_list):
    """ 
    save training summary to disk 
    """
    train_results_df = pd.DataFrame(results_list, columns=columns_list)
    train_results_df['ephoch_index'] = np.arange(train_results_df.shape[0])
    csv_path = change_file_ending(model_path, '.csv')
    train_results_df.to_csv(csv_path)
    return train_results_df



def optimization_improve_checker(best_model_score, current_val, max_opt, model,best_model_wts,
                                 model_path, patience, is_worm_up):
    """ 
    check if model was improved in this ephoc 
    """
    # maximization\minimization and validation score was improved 
    if (max_opt and current_val >= best_model_score) or (not max_opt and current_val <= best_model_score):
        # update weights 
        best_model_wts = model.state_dict()
        if model_path != '':
            # save model 
            torch.save(model, model_path)

        if current_val == best_model_score:
            # patience -= 0.5
            pass
        else:
            patience = 0
        best_model_score = current_val
    if is_worm_up:  
        patience += 1
    return best_model_wts, best_model_score, patience

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
        
def write_final_results_to_tensorboard(device, epoch, model, tb_writer, train_loader, val_loader):
    if not tb_writer is None and train_loader.dataset.learning_type == 'supervised':
        # add pr curves to tensor board
        add_pr_curves_to_tensorboard(model, val_loader,
                                     device,
                                     tb_writer, epoch, num_classes=train_loader.dataset.amount_of_class)

        add_wrong_prediction_to_tensorboard(model, val_loader, device, tb_writer,
                                            1, tag='Wrong_Predections', max_images=50)
        

def schedular_step(scheduler, val_classification_loss):
    """ 
    apply schedular step
    """
    if not scheduler is None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_classification_loss)
        else:
            scheduler.step()
            
def update_ephoch_result(max_opt, val_classification_loss, val_f1_score):
    """ 
    in maximization our focous on accuracy while in minimization on loss (ssl)
    """
    if max_opt:
        current_val = val_f1_score
    else:
        current_val = val_classification_loss
    return current_val


        
def main(model, student, optimizer, classification_criterion, ranking_criterion, accuracy_metric, perm_creterion,
             train_loader, val_loader, num_epochs, device, tb_writer = None, 
         scheduler = None, model_path = '', max_opt = True, scheduler_worm_up = None):
    
    def worm_lr_lambda(epoch, end_lr, worn_up_long = 20):
        factor = 0.1
        if epoch < worn_up_long:
            return end_lr*factor + (end_lr - end_lr*factor) * epoch / worn_up_long
        return end_lr
    
    # initilized list 
    accuracy_train_list = []
    accuracy_validation_list = []
    loss_train_list = []
    loss_validation_list = []
    
    initial_lr  = optimizer.param_groups[0]['lr']
    # set optimizer for sigma parameters     
    if hasattr(model, 'sigma'):
        optimizer_sigma = torch.optim.Adam([model.sigma], lr=4e-4)
    else:
        optimizer_sigma =   None
    
    # generate summary columns 
    columns_list = generate_summary_columns(model)
    results_list = []
    best_model_score = initilizied_best_result(max_opt)
    
    # set early stoping condition     
    max_patience, patience = set_early_stoping_parameters()
    best_model_wts = None
    # model.is_worm_up = False
    
    is_worm_up = (best_model_score >= model.worm_up and max_opt) or (best_model_score <= model.worm_up and not max_opt)
    model.is_worm_up = is_worm_up
    # run for predefined amount of ephoch
    for epoch in range(num_epochs):
        
        # worm up if needed learning rate 
        # if not model.is_worm_up:
        #     is_worm_up = (best_model_score >= model.worm_up and max_opt) or (best_model_score <= model.worm_up and not max_opt)
        #     model.is_worm_up = is_worm_up
        # if not is_worm_up and 0: 
        #     worn_up_long = 20
        #     optimizer.param_groups[0]['lr'] = worm_lr_lambda(epoch, initial_lr, worn_up_long = worn_up_long)
            
        # train
        model.epoch = epoch
        # run train step 
        train_accuracy, train_f1_score, train_classification_loss, \
        train_perm_classification_loss, train_f1_perm_score = \
            train(model, student, optimizer, optimizer_sigma, classification_criterion, 
                  ranking_criterion, accuracy_metric, perm_creterion, train_loader, device,
                  scheduler=scheduler, epoch = epoch, num_epochs=num_epochs )
        
        
        # run validation step  
        val_accuracy, val_f1_score, \
        val_classification_loss, val_perm_classification_loss, \
        val_f1_perm_score = eval_model(model, student, classification_criterion,
                                         ranking_criterion, accuracy_metric, perm_creterion,
                                         val_loader, device)
        
        
        
        # update model score base optimization task 
        current_val = update_ephoch_result(max_opt, val_classification_loss, val_accuracy)
        
        # suffle permutation per image for image1\2
        random.shuffle(train_loader.dataset.perm_order_list)
        random.shuffle(train_loader.dataset.perm_order_list2)
        
        # print current results
        print_epoch_results(epoch, model, train_accuracy, train_classification_loss, train_f1_perm_score,
                                train_f1_score, train_perm_classification_loss, val_accuracy, val_classification_loss,
                                val_f1_perm_score, val_f1_score, val_perm_classification_loss)

        
        # add apoch results 
        results_list = add_apoch_results(model, results_list, train_accuracy, train_classification_loss, train_f1_perm_score,
                              train_f1_score, train_perm_classification_loss, val_accuracy, val_classification_loss,
                              val_f1_perm_score, val_f1_score, val_perm_classification_loss)
        
        
        # save model result 
        train_results_df = save_training_summary_results(columns_list, model_path, results_list)

        # worm up step 
        
        # optimizer.param_groups[0]['lr'] = initial_lr
        best_model_wts, best_model_score, patience  = \
             optimization_improve_checker(best_model_score, current_val, max_opt, model,best_model_wts,
                                         model_path, patience, model.is_worm_up)
             
        # if is not worm up check if worm in last epoch
        if not model.is_worm_up:
            # check 
            is_worm_up = (best_model_score >= model.worm_up and max_opt) or (best_model_score <= model.worm_up and not max_opt)
        
        # check if there is change in worm up
        if is_worm_up != model.is_worm_up:
            best_model_score = initilizied_best_result(max_opt)
            
            # update if model is wormed
            model.is_worm_up = is_worm_up
            
        if is_worm_up: 
            schedular_step(scheduler, val_classification_loss)
        # is passs early stoping condition 
        if patience>max_patience:
          declare_early_stopping_condition(max_patience, model)
          # if model.learning_type == 'supervised':
          #     print(f'validation f1 score does not improve for {max_patience} epoch, therefore optimization is stop due early stoping condition')
          # else:
          #     print(f'validation total loss score does not improve for {max_patience} epoch, therefore optimization is stop due early stoping condition')
          break
        # tensor board     
        if not tb_writer is None: 
            # add scalar (loss/accuracy) to tensorboard
            # write_scalar_2_tensorboard(epoch, tb_writer, train_accuracy, train_classification_loss, train_f1_score,
            #                                 val_accuracy, val_classification_loss, val_f1_score)
            pass
            # tb_writer.add_scalar('Loss/Loss', val_classification_loss, epoch)
            # tb_writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
            # tb_writer.add_scalar('F_score/Validation', val_f1_score, epoch)

            # # add scalars (loss/accuracy) to tensorboard
            # tb_writer.add_scalars('Loss/train-val', {'train': train_classification_loss, 
            #                                          'validation': val_classification_loss}, epoch)
            # tb_writer.add_scalars('Accuracy/train-val', {'train': train_accuracy, 
            #                                              'validation': val_accuracy}, epoch)
            
            # tb_writer.add_scalars('F_score/train-val', {'train': train_f1_score, 
            #                                              'validation': val_f1_score}, epoch)
            
            # adding model weights to tensorboard as histogram
            # add_model_weights_as_histogram(model, tb_writer, epoch)
        
       
            
    
    if not tb_writer is None and train_loader.dataset.learning_type == 'supervised':
        # add pr curves to tensor board
        # add_pr_curves_to_tensorboard(model, val_loader, 
        #                              device, 
        #                              tb_writer, epoch, num_classes = train_loader.dataset.amount_of_class)
      
        # add_wrong_prediction_to_tensorboard(model, val_loader, device, tb_writer, 
        #                                     1, tag='Wrong_Predections', max_images=50)
        pass
    # save model results 
    train_results_df = save_training_summary_results(columns_list, model_path, results_list)
    
    # load best model
    model.load_state_dict(best_model_wts)
    
    
    return train_results_df


