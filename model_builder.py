import torchvision.models as models
import torch.nn as nn
from torchsummary import summary
import torch 
import numpy as np
import copy


        
        
def update_moving_average(ema_updater, student_model, teacher_model):
    # max_update_size = list(student_model.parameters()).__len__()-1
    for idx, (teacher_params, student_params) in enumerate(zip(teacher_model.named_parameters(), student_model.named_parameters())):
            # print(idx)
            # if idx == 650:
            #     a=5
            # print(teacher_params[0])
            # print(teacher_params[1].shape)

            # print(student_params[0])
            # print(student_params[1].shape)

            # get current weights
            old_weight, up_weight = student_params[1].data, teacher_params[1].data
            
            # update student weights
            student_params[1].data = ema_updater.update_average(old_weight, up_weight)
            # print(student_params[1].requires_grad)
        
def model_sellection(model_bank, model_name = 'efficientnet_v2_m', weights = 'IMAGENET1K_V1' ):
    """
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
    # for param in backbone.named_parameters():
    #      debug= True
    #      if debug:
    #          print(param[0])
                          
    return backbone

def print_grad(model):
    for idx, param in enumerate(model.named_parameters()):
         if not param[1].grad is None:
             max_grad = torch.max(torch.abs(param[1].grad))
             print(param[0] + ' grad -- '+str(max_grad.item()))

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

    for param in model.named_parameters():
         debug= False
         if debug:
             print(param[0])
         # TODO! remove features.7.4.block.3
         if (param[0].find('features.8') !=-1 or param[0].find('bn') !=-1) :

         # if (param[0].find('features.8') !=-1   or param[0].find('features.7.4.block.3') !=-1  or param[0].find('bn') !=-1) :
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
        
        
    for param in model.named_parameters():
         debug= False
         if debug:
             print(param[0])
         # if param[0].find(last_layer_name) !=-1 :

         # if (param[0].find('layer4.2') !=-1 or  param[0].find('bn') !=-1):
         if param[0].find('layer4') !=-1 :
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
    model_layers_names = [n for n, _ in model.named_children()]
    return model_layers_names


def get_last_layer_input_size(model, model_layers_names):
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
    model_layers_names = get_model_layers_names(model)
    input_tensor = torch.rand((1,image_dim[0],image_dim[1], image_dim[2] ))
    for i_layer in model_layers_names[0:-1]:
        backbone_layer = getattr(model, i_layer)
        input_tensor = backbone_layer(input_tensor)
                
    return input_tensor.data.shape

def get_model_output_shape(model, image_dim):
    model_layers_names = get_model_layers_names(model)
    input_tensor = torch.rand((1,image_dim[0],image_dim[1], image_dim[2] ))
    for i_layer in model_layers_names[0::]:
        backbone_layer = getattr(model, i_layer)
        input_tensor = backbone_layer(input_tensor)
                
    return input_tensor.data.shape



def update_classifier_head(backbone, image_dim, num_classes, model_name = 'efficientnet_v2_m'):
    # get shape after backbone 
    shape_size = get_output_shape(backbone, image_dim)
    flatten_size = np.prod(list(shape_size))
    
    # set regression head
    hidden_size = int(flatten_size*0.75)
    
    if hidden_size//4 < num_classes:
        assert False, 'needed to change classifier hidden size due amount of class'
   
    HEAD = torch.nn.Sequential(
                                nn.Dropout(p=0.3),
                                nn.Linear(flatten_size, hidden_size),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.25),
                                nn.Linear(hidden_size, hidden_size//4),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.25),
                                nn.Linear(hidden_size//4,num_classes)
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
                     copy_weights = True):
    
    amount_of_patch = training_configuration.amount_of_patch
    
    if training_configuration.learning_type == 'supervised':
        student = None
    else:
        student  =  CNN(training_configuration, 
                        num_classes = amount_of_class,
                        image_dim = (3,image_dim, image_dim),
                        freeze_all = freeze_all,
                        model_name = model_name,
                        weights=weights, unfreeze = unfreeze)  
        if copy_weights:
            student.backbone.load_state_dict(teacher.backbone.state_dict())
        
        
        last_layer_name = get_model_layers_names(student.backbone)[-1]
        projection_layer = getattr(student.backbone, last_layer_name)
        project_layer_list = get_model_layers_names(projection_layer)
        amount_of_layers = project_layer_list.__len__()

        new_projection_layer = nn.Sequential(*list(projection_layer.children())[0:amount_of_layers-4])
        setattr(student.backbone, last_layer_name, new_projection_layer)
        
        
        
        # last_layer_name = get_model_layers_names(student.backbone)[-1]
        # projection_layer = getattr(student.backbone, last_layer_name)
        # project_layer_list = get_model_layers_names(projection_layer)
        # amount_of_layers = project_layer_list.__len__()
        freeze_all_layers(student)

        # student.REPRESENTATION_HEAD = nn.Identity()

        # new_projection_layer = nn.Sequential(*list(projection_layer.children())[0:amount_of_layers-3])
        # setattr(student.backbone, last_layer_name, new_projection_layer)
        
        
        
        update_moving_average(teacher.student_ema_updater, student, teacher)
        freeze_all_layers(student)
 
                
                


    return student



def update_representation_head(backbone, image_dim, num_classes, \
                               model_name = 'efficientnet', \
                               amount_of_patch = 25, hidden_size=512,max_allowed_permutation = 1000):
    # get shape after backbone 
    shape_size = get_output_shape(backbone, image_dim)
    flatten_size = np.prod(list(shape_size))
    
    # set regression head
    hidden_size = int(hidden_size)
    hidden2 = 2056
    if hidden_size//4 < num_classes:
        assert False, 'needed to change classifier hidden size due amount of class'
    
    # HEAD = torch.nn.Sequential(
    #                             nn.Dropout(p=0.3),
    #                             nn.Linear(flatten_size, int(flatten_size*0.75)),
    #                             nn.ReLU(inplace=True),
    #                             nn.Dropout(p=0.25),
    #                             nn.Linear(int(flatten_size*0.75), hidden_size),
    #                             nn.ReLU(inplace=True),
    #                             nn.Dropout(p=0.25),
    #                             nn.Linear(hidden_size, hidden_size), 
    #                             nn.ReLU(inplace=True)

    #                             )
    
    """
    BRD BATCH NORM --> RELU --> DROPOUT
    """
    # BODY = torch.nn.Sequential(
    #                             nn.Dropout(p=0),
    #                             nn.Linear(flatten_size, hidden2),
    #                             nn.BatchNorm1d(hidden2),
    #                             nn.ReLU(inplace=True),
    #                             nn.Dropout(p=0),
    #                             nn.Linear(hidden2, hidden_size),
    #                             nn.BatchNorm1d(hidden_size),
    #                             nn.ReLU(inplace=True),
    #                             nn.Dropout(p=0),            
    #                             nn.Linear(hidden_size, hidden_size))

    BODY = torch.nn.Sequential(
                                nn.Flatten()
                                )
                                
    
                                
    
    REPRESENTATION_HEAD = torch.nn.Sequential(  
                                                nn.Flatten(),
                                                nn.Dropout(p=0),
                                                nn.Linear(flatten_size, hidden2),
                                                nn.BatchNorm1d(hidden2),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(p=0),
                                                nn.Linear(hidden2, hidden_size),
                                                nn.BatchNorm1d(hidden_size),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(p=0),            
                                                nn.Linear(hidden_size, hidden_size)
                                                )
    grid_size = int(amount_of_patch**0.5)
    prem_hidden = flatten_size
    
    # PERM_HEAD = torch.nn.Sequential(nn.Dropout(p=0.3),
    #                                 nn.Linear(flatten_size, hidden_size),
    #                                 nn.BatchNorm1d(hidden_size),
    #                                 nn.ReLU(inplace=True),
    #                                 nn.Dropout(p=0.25),
    #                                 nn.Linear(hidden_size, amount_of_patch)
    #                                 )
    
    prem_hidden2 = int(prem_hidden*0.75)
    # PERM_HEAD = torch.nn.Sequential(nn.Dropout(p=0),
    #                                 nn.Linear(prem_hidden, prem_hidden2),
    #                                 nn.BatchNorm1d(prem_hidden2),
    #                                 nn.ReLU(inplace=True),
    #                                 nn.Dropout(p=0),
    #                                 nn.Linear(prem_hidden2, prem_hidden2),
    #                                 nn.BatchNorm1d(prem_hidden2),
    #                                 nn.ReLU(inplace=True),
    #                                 nn.Dropout(p=0),
    #                                 nn.Linear(prem_hidden2, prem_hidden2),
    #                                 nn.BatchNorm1d(prem_hidden2),
    #                                 nn.ReLU(inplace=True),
    #                                 nn.Dropout(p=0),
    #                                 nn.Linear(prem_hidden2,amount_of_patch))
    
    # prem_hidden = 512
    PERM_HEAD = torch.nn.Sequential(
                                    # nn.Linear(prem_hidden, prem_hidden2),
                                    # nn.BatchNorm1d(prem_hidden2),
                                    # nn.ReLU(inplace=True),
                                    # nn.Dropout(p=0),
                                    # nn.Linear(prem_hidden2, prem_hidden2),
                                    # nn.BatchNorm1d(prem_hidden2),
                                    # nn.ReLU(inplace=True),
                                    # nn.Dropout(p=0),
                                    # nn.Linear(prem_hidden2, prem_hidden2),
                                    nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Flatten(),
                                    nn.Dropout(p=0),
                                    nn.Linear(prem_hidden, prem_hidden//2),
                                    nn.BatchNorm1d(prem_hidden//2),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0),
                                    nn.Linear(prem_hidden//2, prem_hidden//4),
                                    nn.BatchNorm1d(prem_hidden//4),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0),
                                    nn.Linear(prem_hidden//4,amount_of_patch))
    
    # PERM_HEAD = torch.nn.Sequential(
    #                                 # nn.Linear(prem_hidden, prem_hidden2),
    #                                 # nn.BatchNorm1d(prem_hidden2),
    #                                 # nn.ReLU(inplace=True),
    #                                 # nn.Dropout(p=0),
    #                                 # nn.Linear(prem_hidden2, prem_hidden2),
    #                                 # nn.BatchNorm1d(prem_hidden2),
    #                                 # nn.ReLU(inplace=True),
    #                                 # nn.Dropout(p=0),
    #                                 # nn.Linear(prem_hidden2, prem_hidden2),
    #                                 nn.AdaptiveAvgPool2d((1, 1)),
    #                                 nn.Flatten(),
    #                                 # nn.Dropout(p=0),
    #                                 # nn.Linear(prem_hidden, prem_hidden//2),
    #                                 nn.BatchNorm1d(prem_hidden),
    #                                 nn.ReLU(inplace=True),
    #                                 nn.Dropout(p=0),
    #                                 # nn.Linear(prem_hidden//2, prem_hidden//4),
    #                                 # nn.BatchNorm1d(prem_hidden//4),
    #                                 # nn.ReLU(inplace=True),
    #                                 # nn.Dropout(p=0),
    #                                 nn.Linear(prem_hidden, amount_of_patch))
    
    
    PERM_LABEL_HEAD = torch.nn.Sequential(
                                    # nn.Linear(prem_hidden, prem_hidden2),
                                    # nn.BatchNorm1d(prem_hidden2),
                                    # nn.ReLU(inplace=True),
                                    # nn.Dropout(p=0),
                                    # nn.Linear(prem_hidden2, prem_hidden2),
                                    # nn.BatchNorm1d(prem_hidden2),
                                    # nn.ReLU(inplace=True),
                                    # nn.Dropout(p=0),
                                    # nn.Linear(prem_hidden2, prem_hidden2),
                                    nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Flatten(),
                                    nn.Dropout(p=0),
                                    nn.Linear(prem_hidden, prem_hidden//2),
                                    nn.BatchNorm1d(prem_hidden//2),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0),
                                    nn.Linear(prem_hidden//2, prem_hidden//4),
                                    nn.BatchNorm1d(prem_hidden//4),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0),
                                    nn.Linear(prem_hidden//4,max_allowed_permutation))
    
    # PERM_LABEL_HEAD = torch.nn.Sequential(
    #                                 # nn.Linear(prem_hidden, prem_hidden2),
    #                                 # nn.BatchNorm1d(prem_hidden2),
    #                                 # nn.ReLU(inplace=True),
    #                                 # nn.Dropout(p=0),
    #                                 # nn.Linear(prem_hidden2, prem_hidden2),
    #                                 # nn.BatchNorm1d(prem_hidden2),
    #                                 # nn.ReLU(inplace=True),
    #                                 # nn.Dropout(p=0),
    #                                 # nn.Linear(prem_hidden2, prem_hidden2),
    #                                 nn.AdaptiveAvgPool2d((1, 1)),
    #                                 nn.Flatten(),
    #                                 # nn.Dropout(p=0),
    #                                 # nn.Linear(prem_hidden, prem_hidden//2),
    #                                 nn.BatchNorm1d(prem_hidden),
    #                                 nn.ReLU(inplace=True),
    #                                 nn.Dropout(p=0),
    #                                 # nn.Linear(prem_hidden//2, prem_hidden//4),
    #                                 # nn.BatchNorm1d(prem_hidden//4),
    #                                 # nn.ReLU(inplace=True),
    #                                 # nn.Dropout(p=0),
    #                                 nn.Linear(prem_hidden, max_allowed_permutation))
    
    
    # freeze_all_layers(PERM_HEAD)
    # nn.Tanh()
    
    # m = nn.AdaptiveAvgPool2d((1, 1))
    # input = torch.randn(1, 64, 8, 9)
    # m(input).shape
    
    if model_name.find('resnet') != -1:
        # set classification head 
        backbone.fc = BODY
    elif model_name.find('efficientnet') != -1:
        # set classification head 
        backbone.classifier = BODY
    return PERM_HEAD, REPRESENTATION_HEAD, PERM_LABEL_HEAD
        
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.initial_beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

        
        
class SSLMODEL(nn.Module):
    def __init__(self, model,
                 num_classes, 
                 image_dim=(3,224,224), 
                 freeze_all=False,
                 model_name='efficientnet_v2_m',
                 unfreeze=False):
      
        super(SSLMODEL, self).__init__()
        
        ssl_model = copy.deepcopy(model.backbone)
        ssl_model.learning_type = 'supervised'
        # del ssl_model.PERM_HEAD
        # del ssl_model.REPRESENTATION_HEAD
        # del ssl_model.PERM_LABEL_HEAD
        model_bank = ['resnet18','resnet34', 'resnet50', 
                      'resnet152', 'efficientnet_v2_m', 
                      'efficientnet_v2_s', 'efficientnet_v2_l']
        # model_name = 'efficientnet_v2_m'
        
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
            
            # for param in backbone.named_parameters():
            #       debug= True
            #       if debug:
            #           print(param[0])
        
        update_classifier_head(ssl_model, image_dim, num_classes, model_name = model_name )
        self.learning_type = 'supervised'
        self.ssl_model = ssl_model
        moving_average_decay = 0.99
        use_momentum = True
        self.use_momentum = use_momentum
        self.target_encoder = None
        self.student_ema_updater = EMA(moving_average_decay)
        

    def forward(self, images):
        
        classification_pred = self.ssl_model(images)

        return classification_pred
    
    
def forward_using_loop(model, data):
    layer_input = data
    for layer_idx , layer  in enumerate(model.backbone.children()):
        model_layers_names = get_model_layers_names(layer)
        # print(len(model_layers_names))
        # print(model_layers_names)
        # print(layer_idx)
        if layer_idx == 7:
            a=5
        if len(model_layers_names) == 0:
                layer_output  = layer(layer_input)
                if layer_idx == 7 :
                    gemotric_output = layer_output
                layer_input = layer_output
        else:
    
            for sub_layer_idx , sub_layer  in enumerate(layer.children()):
    
                layer_output  = sub_layer(layer_input)
                if layer_idx == 7 and sub_layer_idx == 0:
                    gemotric_output = layer_output
                layer_input = layer_output
    return layer_output, gemotric_output



class CNN(nn.Module):
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

        
        model_bank = ['resnet18','resnet34', 'resnet50', 'resnet152',
                      'efficientnet_v2_m', 'efficientnet_v2_s', 'efficientnet_v2_l']
        # model_name = 'resnet50'
        # sellecting backbone from torchvision models
        backbone = model_sellection(model_bank, 
                                    model_name=model_name,
                                    weights=weights)
        
       
        # dummy_array =  torch.rand((1,image_dim[0],image_dim[1], image_dim[2] ))
        
        # freeze all except batch norm layer 
        freeze_backbone_layers(backbone, 
                               model_bank, 
                               model_name=model_name, 
                               freeze_all=freeze_all,
                               unfreeze=unfreeze)
        
        def getge_output_shape(model, image_dim):
            return model.avgpool(torch.rand((1,image_dim[0],image_dim[1], image_dim[2] ))).data.shape
        # get shapes 
        channel, height, width = image_dim
                
        
        model_layers_names = get_model_layers_names(backbone)
        
        # print backbone summary
        debug = False
        if debug:    
            summary(backbone, (channel, height, width))
            
    
        if learning_type== 'supervised':
            update_classifier_head(backbone, image_dim, num_classes, model_name = model_name)
            PERM_HEAD = REPRESENTATION_HEAD = PERM_LABEL_HEAD = None
        else:
            PERM_HEAD, REPRESENTATION_HEAD, PERM_LABEL_HEAD = update_representation_head(backbone, image_dim, num_classes,
                                                                        model_name = model_name,
                                                                        amount_of_patch = amount_of_patch,
                                                                        hidden_size=hidden_size, max_allowed_permutation = max_allowed_permutation)
            # PERM_HEAD, REPRESENTATION_HEAD = None, None
        
        # for param in PERM_HEAD.named_parameters():
        #     print(param[1].requires_grad)
             
        self.sigma = nn.Parameter(torch.ones(3))
        # self.sigma.data[1:3] = 0.5
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
            # projection_output = self.backbone(images)
            # geometric_output = self.backbone.features[0](images)
            projection_output, geometric_output = forward_using_loop(self, images)
            
            
            # projection_output = self.backbone(images)
            # perm_pred = torch.rand(images.shape[0], 25, requires_grad=True)
            # perm_pred = self.PERM_HEAD(geometric_output.clone())
            
            perm_label_pred = self.PERM_LABEL_HEAD(geometric_output.clone())
            perm_pred = self.PERM_HEAD(geometric_output.clone())

            
            representation_pred = self.REPRESENTATION_HEAD(projection_output.clone())
            del projection_output
            # del projection_output, geometric_output
# 
            # representation_pred = self.backbone(images)
            # return representation_pred, perm_pred
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
         if 0 :
         # if (param[0].find(model_layers_names[-1]) !=-1  or  param[0].find('bn') !=-1) or param[0].find('features.7') !=-1:
             # print(param[0])
             param[1].requires_grad = True
         else:
             param[1].requires_grad = False 








