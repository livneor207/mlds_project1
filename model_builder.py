import torchvision.models as models
import torch.nn as nn
from torchsummary import summary
import torch 
import numpy as np
import copy
import os 



def load_model(model_load_path, learning_type = 'supervised', device = 'cpu'):
    """
    

    Parameters
    ----------
    model_load_path : model path to load 
    learning_type : type of learning could be supervised\ssl
    device : cpu\gpu
    Returns
    -------
    model : loaded model after load to device 
    """
    if os.path.exists(model_load_path):
        model = torch.load(model_load_path, map_location='cpu')
        for name, param in model.named_parameters():
            if param.device.type != 'cpu':
                param.to('cpu')
        if learning_type  != 'supervised':
            print('model sigma')
            print(model.sigma)
    else:
        print('can not load model because file is not exists')
    return model


def update_moving_average(ema_updater, student_model, teacher_model):
    """
    

    Parameters
    ----------
    ema_updater : method for EMA upadte methood
    student_model : offline model (freeze)
    teacher_model : online model (learnable)

    Returns
    -------
    model after student weight are updated using EMA update 

    """
    # this loop run on all parameters
    for idx, (teacher_params, teacher_module, student_params, student_module) in enumerate(zip(teacher_model.named_parameters(), teacher_model.modules() , student_model.named_parameters(), student_model.modules())):
            # print(teacher_params)
            # print(student_params)            
            # print(idx)
            
            
            # get paramters name
            teacher_parameter_name = teacher_params[0]
            student_parameter_name = student_params[0]
            
            # get parameter weights
            old_weight, up_weight = student_params[1], teacher_params[1]

            # handle batch norm layers
            if isinstance(student_module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # get online model mean & var (update as running average)
                teacher_mean =  teacher_module.running_mean
                teacher_var =   teacher_module.running_var
                # get offline model mean & var (update as running average)
                student_mean =  student_module.running_mean
                student_var =  student_module.running_var
                
                # update using moving avarage
                if up_weight.requires_grad:
                    student_module.running_mean.data = ema_updater.update_average(student_mean.data, teacher_mean.data)
                    student_module.running_var.data = ema_updater.update_average(student_var.data, teacher_var.data)
                # copy mean& var
                else:
                    student_module.running_mean.data = teacher_mean.data
                    student_module.running_var.data = teacher_var.data
                    
            # update using moving avarage
            if up_weight.requires_grad:
                student_params[1].data = ema_updater.update_average(old_weight.data, up_weight.data)
            else:
                student_params[1].data = up_weight.data
 
                
def model_sellection(model_bank, model_name = 'efficientnet_v2_m', weights = 'IMAGENET1K_V1' ):
    """
    load pretrained models base there names
    the only suported model is resnet50
    https://pytorch.org/vision/stable/models.html
    """

    if model_name not in model_bank:
        assert False, 'model is not defined'
    if model_name == 'resnet18':
        backbone = models.resnet18(weights=weights)
    elif model_name == 'resnet34':
        backbone = models.resnet34(weights=weights)
    elif model_name == 'resnet50':
        backbone = models.resnet50(weights=weights)
    elif model_name == 'resnet152':
        # defult model - resnet18
        backbone = models.resnet152(weights=weights)
    elif model_name == 'efficientnet_v2_s':
        backbone = models.efficientnet_v2_s(weights=weights)
    elif model_name == 'efficientnet_v2_m':
        backbone = models.efficientnet_v2_m(weights=weights)
    elif model_name == 'efficientnet_v2_l':
        backbone = models.efficientnet_v2_l(weights=weights)

                          
    return backbone

def print_grad(model):
    """
    print model grads
    """
    for idx, param in enumerate(model.named_parameters()):
         if not param[1].grad is None:
             max_grad = torch.max(torch.abs(param[1].grad))
             print(param[0] + ' grad -- '+str(max_grad.item()))

def collect_grads(model):
    """ 
    collect model grads and calculate the precentile 95 in order clip grad
    """
    grad_list = []
    for idx, param in enumerate(model.named_parameters()):
         if not param[1].grad is None and param[0] not in ['sigma']:
             max_grad = torch.max(torch.abs(param[1].grad))
             grad_list.append(max_grad.item())
             
    percentile_95 = np.percentile(grad_list, 95)
    max_norm = percentile_95 * 1.25
    return max_norm

def freeze_efficientnet_layers(model, model_name = 'efficientnet_v2_m'):
        
    """
    freeze by defult all layer
    otherwise manully freeze layers 
    """
    last_layer = ''
    if model_name == 'efficientnet_v2_m':
        last_layer = 'features.8'
    elif model_name == 'efficientnet_v2_s':
        last_layer = 'features.7'
    
    for idx, (param, param_module) in enumerate(zip(model.named_parameters(), model.modules())):

    # for param in model.named_parameters():
         debug= False
         if debug:
             print(param[0])
         if (param[0].find('features.8') !=-1 or isinstance(param_module, (nn.BatchNorm2d, nn.BatchNorm1d))) :

            param[1].requires_grad = True
         else:
            param[1].requires_grad = False

def freeze_resnet_layers(model, model_name = 'resnet50'):        
    """
    freeze by defult all layer
    otherwise manully freeze layers 
    """
    if int(model_name.split('resnet')[1])>40:
        last_layer_name = 'layer4.2'
    else:
        last_layer_name = 'layer4.1'
    
    if int(model_name.split('resnet')[1])>40:
        last_layer_name2 = 'layer2.2'
    else:
        last_layer_name2 = 'layer2.1'
        
        
    for idx, (param, param_module) in enumerate(zip(model.named_parameters(), model.modules())):
         debug= False
         if debug:
             print(param[0])

         if param[0].find('layer4') !=-1 :
         # if (param[0].find('layer4.2') !=-1 or  isinstance(param_module, (nn.BatchNorm2d, nn.BatchNorm1d))):
            param[1].requires_grad = True
         else:
            param[1].requires_grad = False
      


def freeze_all_layers(model):        
    """
    freeze by defult all layer
    otherwise manully freeze layers 
    """
    for param in model.named_parameters():
         debug= False
         if debug:
             print(param[0])
         param[1].requires_grad = False
def unfreeze_all_layers(model):        
    """
    freeze by defult all layer
    otherwise manully freeze layers 
    """
    for param in model.named_parameters():
         debug= False
         if debug:
             print(param[0])
         param[1].requires_grad = True      
            
def freeze_backbone_layers(model, model_bank, model_name = 'resnet34', freeze_all = True, unfreeze = False):        
    """
    freeze by defult all layer
    otherwise manully freeze layers 
    """
    if unfreeze:
      pass
    elif model_name not in model_bank:
        assert False, 'model is not defined'
    elif freeze_all:
        freeze_all_layers(model)
    elif model_name.find('resnet') != -1:
        freeze_resnet_layers(model, model_name)
    elif model_name.find('efficientnet') != -1:
        freeze_efficientnet_layers(model, model_name)
             
def get_model_layers_names(model):
    """ 
    get sublayer names list
    """
    model_layers_names = [n for n, _ in model.named_children()]
    return model_layers_names


def get_last_layer_input_size(model, model_layers_names):
    """ 
    get the shape last layer input shape size 
    """
    last_layer_name =  model_layers_names[-2]
    last_layer = getattr(model, last_layer_name)
    last_layer_layers  = get_model_layers_names(last_layer)
    last_layer_name = ''
    for layer_name in last_layer_layers:
        if layer_name.find('Linear')!=-1:
            last_layer_name = layer_name
            break
        
    if last_layer_name == '':
        assert False, 'no Linear layer in last layer'
    else:
        last_layer = getattr(last_layer, last_layer_name)

    model_last_layer_input_size = last_layer.weight.shape[0]
    return model_last_layer_input_size


def get_output_shape(model, image_dim):
    """ 
    get outsize base input size 
    """
    model_layers_names = get_model_layers_names(model)
    input_tensor = torch.rand((1,image_dim[0],image_dim[1], image_dim[2] ))
    for i_layer in model_layers_names[0:-1]:
        backbone_layer = getattr(model, i_layer)
        input_tensor = backbone_layer(input_tensor)
                
    return input_tensor.data.shape

def get_model_output_shape(model, image_dim):
    """ 
    get outsize base input size 
    """
    model_layers_names = get_model_layers_names(model)
    input_tensor = torch.rand((1,image_dim[0],image_dim[1], image_dim[2] ))
    for i_layer in model_layers_names[0::]:
        backbone_layer = getattr(model, i_layer)
        input_tensor = backbone_layer(input_tensor)
                
    return input_tensor.data.shape



def update_classifier_head(backbone, image_dim, num_classes, model_name = 'efficientnet_v2_m'):
    """ 
    append to model feature exractor mlp layer for classification task
    """
    # get shape after backbone 
    shape_size = get_output_shape(backbone, image_dim)
    flatten_size = np.prod(list(shape_size))
    

    if hidden_size//4 < num_classes:
        assert False, 'needed to change classifier hidden size due amount of class'
   
    HEAD = torch.nn.Sequential(
                                nn.Dropout(p=0.25),
                                nn.Linear(flatten_size, num_classes)
                                )

    if model_name.find('resnet') != -1:
        # set classification head 
        backbone.fc = HEAD
    elif model_name.find('efficientnet') != -1:
        # set classification head 
        backbone.classifier = HEAD
        
        

        
def generate_student(teacher, training_configuration, image_dim, 
                     amount_of_class, model_name = 'efficientnet',
                     amount_of_patch = 1, freeze_all = False,
                     weights = 'IMAGENET1K_V1', unfreeze = True,
                     copy_weights = True, update_student = True):
    
    """
    generate offline model base online model
    the student model is equivalent to online model less last mlp layer
    """    
    # get amount of patch
    amount_of_patch = training_configuration.amount_of_patch
    
    # generate dumm model
    if training_configuration.learning_type == 'supervised':
        student = None
    # generate model
    else:
        
        # generate duummy resnet50 model
        student  =  CNN(training_configuration, 
                        num_classes = amount_of_class,
                        image_dim = (3,image_dim, image_dim),
                        freeze_all = freeze_all,
                        model_name = model_name,
                        weights=weights, unfreeze = unfreeze)  
        # copy online model weights 
        if copy_weights:
            student.backbone.load_state_dict(teacher.backbone.state_dict())
        
        
        # get representation layer head
        last_layer_name = get_model_layers_names(student.backbone)[-1]
        projection_layer = getattr(student, 'REPRESENTATION_HEAD')
        project_layer_list = get_model_layers_names(projection_layer)
        amount_of_layers = project_layer_list.__len__()
        
        # reduce prediction mlp layer 
        new_projection_layer = nn.Sequential(*list(projection_layer.children())[0:amount_of_layers-6])
        setattr(student, 'REPRESENTATION_HEAD', new_projection_layer)
        student.eval() # model is no trainable
        student.student = True # indicate is student model
        
        # freeze student\offline model
        freeze_all_layers(student)

        # save initial beta 
        old_beta = teacher.student_ema_updater.beta 
        
        # update (forget random initlization)
        if update_student:
            teacher.student_ema_updater.beta = 0 
        # not update
        else:
            teacher.student_ema_updater.beta = 1 
        
        # update all sublayers
        update_moving_average(teacher.student_ema_updater, student.backbone, teacher.backbone)
        update_moving_average(teacher.student_ema_updater, student.REPRESENTATION_HEAD, teacher.REPRESENTATION_HEAD)
        update_moving_average(teacher.student_ema_updater, student.PERM_HEAD, teacher.PERM_HEAD)
        update_moving_average(teacher.student_ema_updater, student.PERM_LABEL_HEAD, teacher.PERM_LABEL_HEAD)
        teacher.student_ema_updater.beta = old_beta

    return student



def update_representation_head(backbone, image_dim, num_classes, \
                               model_name = 'efficientnet', \
                               amount_of_patch = 25, hidden_size=512, max_allowed_permutation=1000):
    # get shape after backbone 
    shape_size = get_output_shape(backbone, image_dim)
    flatten_size = np.prod(list(shape_size))
    if backbone.model_layer == 7:
        flatten_size2 = flatten_size

    else:
        flatten_size2 = flatten_size//2
    # set regression head
    hidden_size = int(hidden_size)
    hidden2 = 2056
    if hidden_size//4 < num_classes:
        assert False, 'needed to change classifier hidden size due amount of class'
    

    
    """
    BRD BATCH NORM --> RELU --> DROPOUT
    """


    BODY = torch.nn.Sequential(
                                nn.Flatten()
                                )
                                
    
    p = 0.2
    REPRESENTATION_HEAD = torch.nn.Sequential(  
                                                # MLP1 Projection
                                                nn.BatchNorm1d(flatten_size),
                                                nn.Linear(flatten_size, hidden2),
                                                nn.BatchNorm1d(hidden2),
                                                nn.ReLU(inplace = True),
                                                nn.Dropout(p=0),
                                                nn.Linear(hidden2, hidden_size),
                                                
                                                # MLP2 Prediction
                                                nn.BatchNorm1d(hidden_size),
                                                nn.Linear(hidden_size, hidden_size),
                                                nn.BatchNorm1d(hidden_size),
                                                nn.ReLU(inplace = True),
                                                nn.Dropout(p=0),
                                                nn.Linear(hidden_size, hidden_size)
                                                )
    grid_size = int(amount_of_patch**0.5)
    prem_hidden = flatten_size2
    prem_hidden2 = int(prem_hidden*0.75)

    
    
    
    
    # MLP3 -  postion embeedding predictions
    PERM_HEAD = torch.nn.Sequential( 
                                    
                                    nn.BatchNorm1d(hidden_size),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.BatchNorm1d(hidden_size),
                                    nn.ReLU(inplace = True),
                                    nn.Dropout(p=0),
                                    nn.Linear(hidden_size, amount_of_patch))
    
    # MLP4 -  permutation predictions head
    PERM_LABEL_HEAD = torch.nn.Sequential(
                                          nn.BatchNorm1d(hidden_size),
                                          nn.Linear(hidden_size, hidden_size),
                                          nn.BatchNorm1d(hidden_size),
                                          nn.ReLU(inplace = True),
                                          nn.Dropout(p=0),
                                          nn.Linear(hidden_size, max_allowed_permutation))
                                    
                  
    if model_name.find('resnet') != -1:
        # set classification head 
        backbone.fc = BODY
    elif model_name.find('efficientnet') != -1:
        # set classification head 
        backbone.classifier = BODY
    return PERM_HEAD, REPRESENTATION_HEAD, PERM_LABEL_HEAD
        
class EMA():
    """
    update class using moving avarge 
    """
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.initial_beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

        
        
class SSLMODEL(nn.Module):
    """
    copy model backbone and add classifier head 
    """
    def __init__(self, model,
                 num_classes, 
                 image_dim=(3,224,224), 
                 freeze_all=False,
                 model_name='efficientnet_v2_m',
                 unfreeze=False):
      
        
        super(SSLMODEL, self).__init__()
        
        # get ssl model backbone
        ssl_model = copy.deepcopy(model.backbone)
        
        # update learning type 
        ssl_model.learning_type = 'supervised'
        
        model_bank = ['resnet18','resnet34', 'resnet50', 
                      'resnet152', 'efficientnet_v2_m', 
                      'efficientnet_v2_s', 'efficientnet_v2_l']
        
        # freeze all except batch norm layer 
        freeze_backbone_layers(ssl_model, model_bank, 
                               model_name = model_name , 
                               freeze_all = freeze_all,
                               unfreeze = unfreeze)
        
        
        def get_output_shape(model, image_dim):
            return model.avgpool(torch.rand((1,image_dim[0],image_dim[1], image_dim[2] ))).data.shape
        # get shapes 
        channel, height, width = image_dim
                
        
        model_layers_names = get_model_layers_names(ssl_model)
        
        # print backbone summary
        debug = False
        if debug:    
            summary(ssl_model, (channel, height, width))
            
        # add classifier
        update_classifier_head(ssl_model, image_dim, num_classes, model_name = model_name )
        self.learning_type = 'supervised'
        self.ssl_model = ssl_model
        moving_average_decay = 0.99
        use_momentum = True
        self.use_momentum = use_momentum
        self.target_encoder = None
        self.student_ema_updater = EMA(moving_average_decay)
        self.worm_up = -1
        

    def forward(self, images):
        
        classification_pred = self.ssl_model(images)

        return classification_pred
    
    
def forward_using_loop(model, data):
    """ 
    forward using loop instead of automatally forward 
    """
    model_sub_layer = model.model_sub_layer
    model_layer = model.model_layer
    layer_input = data
    for layer_idx , layer  in enumerate(model.backbone.children()):
        model_layers_names = get_model_layers_names(layer)
    
        if layer_idx == 7:
            a=5
        amount_of_unit_of_layer =  len(model_layers_names)
        if amount_of_unit_of_layer < 1:
                layer_output  = layer(layer_input)
                # del layer, layer_input
                if layer_idx == model_layer :
                    gemotric_output = layer_output
                layer_input = layer_output
        else:
    
            for sub_layer_idx , sub_layer  in enumerate(layer.children()):
                # print(sub_layer_idx)
                layer_output  = sub_layer(layer_input)
                # del sub_layer, layer_input
                if layer_idx == model_layer and sub_layer_idx == model_sub_layer:
                    gemotric_output = layer_output
                layer_input = layer_output
    if len(gemotric_output.shape) - len(layer_output.shape) == 2:
        gemotric_output = model.avg_2d_pool(gemotric_output)
    return layer_output, gemotric_output



class CNN(nn.Module):
    """ 
    take pretrained model backbone and add classifier head
    """
    def __init__(self, training_configuration, num_classes = 2 , 
                 image_dim = (3,224,224), model_name = 'efficientnet_v2_m',
                 learning_type = 'supervised', amount_of_patch = 1, 
                 hidden_size = 512, freeze_all = False, 
                 weights = 'IMAGENET1K_V1', unfreeze = False):
        
        super(CNN, self).__init__()
        
        learning_type=training_configuration.learning_type
        amount_of_patch = training_configuration.amount_of_patch
        hidden_size=training_configuration.hidden_size
        balance_factor=training_configuration.balance_factor
        balance_factor2=training_configuration.balance_factor2
        max_allowed_permutation = training_configuration.max_allowed_permutation
        moving_average_decay=training_configuration.moving_average_decay
        use_auto_weight = training_configuration.use_auto_weight
        device = training_configuration.device
        model_layer = training_configuration.model_layer
        model_sub_layer = training_configuration.model_sub_layer
        pe_dim = training_configuration.pe_dim
        worm_up = training_configuration.worm_up
        
        
        if learning_type != 'self_supervised':
            worm_up = -1

        model_bank = ['resnet18','resnet34', 'resnet50', 'resnet152',
                      'efficientnet_v2_m', 'efficientnet_v2_s', 'efficientnet_v2_l']
        
        # take backbone
        backbone = model_sellection(model_bank, 
                                    model_name=model_name,
                                    weights=weights)
        
        
        # update from which layer take another output when using forward using loop
        backbone.model_layer = model_layer
        backbone.model_sub_layer = model_sub_layer

        
        # freeze all except batch norm layer 
        freeze_backbone_layers(backbone, 
                               model_bank, 
                               model_name=model_name, 
                               freeze_all=freeze_all,
                               unfreeze=unfreeze)
        
        def getge_output_shape(model, image_dim):
            return model.avgpool(torch.rand((1,image_dim[0],image_dim[1], image_dim[2]))).data.shape
        # get shapes 
        channel, height, width = image_dim
                
        
        model_layers_names = get_model_layers_names(backbone)
        
        # print backbone summary
        debug = False
        if debug:    
            summary(backbone, (channel, height, width))
            
        # for supervied task add classifier head 
        if learning_type== 'supervised':
            update_classifier_head(backbone, image_dim, num_classes, model_name = model_name)
            PERM_HEAD = REPRESENTATION_HEAD = PERM_LABEL_HEAD = None
        # for ssl task - add 3 head to backbone
        else:
            PERM_HEAD, REPRESENTATION_HEAD, PERM_LABEL_HEAD = update_representation_head(backbone, 
                                                                                         image_dim, 
                                                                                         num_classes,
                                                                                         model_name = model_name,
                                                                                         amount_of_patch = pe_dim,
                                                                                         hidden_size=hidden_size, 
                                                                                         max_allowed_permutation = max_allowed_permutation)
        self.avg_2d_pool = torch.nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                               nn.Flatten())
      
        self.model_sub_layer = model_sub_layer
        self.model_layer = model_layer
        self.worm_up = worm_up
        self.pe_dim = pe_dim
        self.device = device
        self.use_auto_weight = use_auto_weight
        self.sigma = nn.Parameter(torch.ones(3))
        # self.sigma.data[1:3] = 0.5
        self.student = False
        self.backbone = backbone
        self.PERM_HEAD = PERM_HEAD
        self.REPRESENTATION_HEAD = REPRESENTATION_HEAD
        self.PERM_LABEL_HEAD = PERM_LABEL_HEAD
        
        moving_average_decay = moving_average_decay
        use_momentum = True
        self.use_momentum = use_momentum
        self.target_encoder = None
        self.student_ema_updater = EMA(moving_average_decay)
        self.learning_type = learning_type
        self.balance_factor = balance_factor
        self.balance_factor2 = balance_factor2

    def forward(self, images):
        if self.learning_type== 'supervised':
            classification_pred = self.backbone(images)
            return classification_pred
        else:
            
            projection_output = self.backbone(images)
            #projection_output, geometric_output = forward_using_loop(self, images)
                           
            representation_pred = self.REPRESENTATION_HEAD(projection_output)
            
            balance_factor = self.balance_factor
            balance_factor2 = self.balance_factor2
            # forward
            if not self.student or balance_factor !=0:
                perm_pred = self.PERM_HEAD(representation_pred)

            else:
                perm_pred = None
            
            # forward 
            if not self.student or balance_factor2 !=0:
                perm_label_pred = self.PERM_LABEL_HEAD(representation_pred)
            else:
                perm_label_pred = None
                
            del projection_output
            return representation_pred, perm_pred, perm_label_pred

    
def freeze_ssl_layers(model):        
    """
    freeze by defult all layer
    otherwise manully freeze layers 
    """
    model_layers_names = get_model_layers_names(model.backbone)

    for param in model.named_parameters():
         debug= False
         if debug: 
             print(param[0])

             param[1].requires_grad = True
         else:
             param[1].requires_grad = False 








