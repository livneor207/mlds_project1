import torchvision.models as models
import torch.nn as nn
from torchsummary import summary
import torch 
import numpy as np
import copy


def update_moving_average(ema_updater, student_model, teacher_model):
    max_update_size = list(student_model.parameters()).__len__()-1
    for idx, (teacher_params, student_params) in enumerate(zip(teacher_model.parameters(), student_model.parameters())):
        try:
            if idx == 650:
                break
            # print(idx)
            # get current weights
            old_weight, up_weight = student_params.data, teacher_params.data
            
            # update student weights
            student_params.data = ema_updater.update_average(old_weight, up_weight)
        except:
            pass
            a=5
def model_sellection(model_bank, model_name = 'efficientnet_v2_m' ):
    """
    https://pytorch.org/vision/stable/models.html
    """

    if model_name not in model_bank:
        assert False, 'model is not defined'
    if model_name == 'resnet34':
        
        backbone = models.resnet34(weights='IMAGENET1K_V1')
    elif model_name == 'resnet50':
        backbone = models.resnet50(weights='IMAGENET1K_V1')
    elif model_name == 'resnet152':
        # defult model - resnet18
        backbone = models.resnet152(weights='IMAGENET1K_V1')
    elif model_name == 'efficientnet_v2_s':
        backbone = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
    elif model_name == 'efficientnet_v2_m':
        backbone = models.efficientnet_v2_m(weights='IMAGENET1K_V1')
    elif model_name == 'efficientnet_v2_l':
        backbone = models.efficientnet_v2_l(weights='IMAGENET1K_V1')
    # for param in backbone.named_parameters():
    #      debug= True
    #      if debug:
    #          print(param[0])
                          
    return backbone



def freeze_efficientnet_layers(model):        
    """
    freeze by defult all layer
    otherwise manully freeze layers 
    """
    for param in model.named_parameters():
         debug= False
         if debug:
             print(param[0])
        
         if (param[0].find('features.7') !=-1 or  param[0].find('bn') !=-1) :
            param[1].requires_grad = True
         else:
            param[1].requires_grad = False

def freeze_resnet_layers(model):        
    """
    freeze by defult all layer
    otherwise manully freeze layers 
    """
    for param in model.named_parameters():
         debug= False
         if debug:
             print(param[0])
        
         if (param[0].find('layer4') !=-1 or  param[0].find('bn') !=-1) :
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
            
            
def freeze_backbone_layers(model, model_bank, model_name = 'resnet34', freeze_all = True):        
    """
    freeze by defult all layer
    otherwise manully freeze layers 
    """
    if model_name not in model_bank:
        assert False, 'model is not defined'
    if freeze_all:
        freeze_all_layers(model)
    elif model_name.find('resnet') != -1:
        freeze_resnet_layers(model)
    elif model_name.find('efficientnet') != -1:
        freeze_efficientnet_layers(model)
             
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



def update_classifier_head(backbone, image_dim, num_classes, model_name = 'efficientnet' ):
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
        
        

        
def generate_student(teacher, training_configuration, image_dim, amount_of_class, model_name = 'efficientnet' ):
    if training_configuration.learning_type == 'supervised':
        student = None
    else:
        student  =  CNN(num_classes = amount_of_class,
                                image_dim = (3,image_dim, image_dim), 
                                learning_type=training_configuration.learning_type)  
        
        student.load_state_dict(teacher.state_dict())
        last_layer_name = get_model_layers_names(student.backbone)[-1]
        projection_layer = getattr(student.backbone, last_layer_name)
        project_layer_list = get_model_layers_names(projection_layer)
        amount_of_layers = project_layer_list.__len__()

        new_projection_layer = nn.Sequential(*list(projection_layer.children())[0:amount_of_layers-3])
        setattr(student.backbone, last_layer_name, new_projection_layer)

        freeze_all_layers(student)
        
        update_moving_average(teacher.student_ema_updater, student, teacher)
                
                
                


    return student



def update_representation_head(backbone, image_dim, num_classes, model_name = 'efficientnet' ):
    # get shape after backbone 
    shape_size = get_output_shape(backbone, image_dim)
    flatten_size = np.prod(list(shape_size))
    
    # set regression head
    hidden_size = int(512)
    
    if hidden_size//4 < num_classes:
        assert False, 'needed to change classifier hidden size due amount of class'
    HEAD = torch.nn.Sequential(
                                nn.Dropout(p=0.3),
                                nn.Linear(flatten_size, int(flatten_size*0.75)),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.25),
                                nn.Linear(int(flatten_size*0.75), hidden_size),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.25),
                                nn.Linear(hidden_size, hidden_size), 
                                nn.ReLU(inplace=True)

                                )
    
    if model_name.find('resnet') != -1:
        # set classification head 
        backbone.fc = HEAD
    elif model_name.find('efficientnet') != -1:
        # set classification head 
        backbone.classifier = HEAD
        
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

        
        

class CNN(nn.Module):
    def __init__(self, num_classes, image_dim = (3,224,224), model_name = 'efficientnet_v2_m', learning_type = 'supervised'):
        super(CNN, self).__init__()
        
        
        model_bank = ['resnet34', 'resnet50', 'resnet152', 'efficientnet_v2_m', 'efficientnet_v2_s', 'efficientnet_v2_l']
        model_name = 'efficientnet_v2_m'
        # sellecting backbone from torchvision models
        backbone = model_sellection(model_bank, model_name = model_name)

        # freeze all except batch norm layer 
        freeze_backbone_layers(backbone, model_bank,  model_name = model_name, freeze_all = True)
        
        
        def get_output_shape(model, image_dim):
            return model.avgpool(torch.rand((1,image_dim[0],image_dim[1], image_dim[2] ))).data.shape
        # get shapes 
        channel, height, width = image_dim
                
        
        model_layers_names = get_model_layers_names(backbone)
        
        # print backbone summary
        debug = False
        if debug:    
            summary(backbone, (channel, height, width))
            
            # for param in backbone.named_parameters():
            #       debug= True
            #       if debug:
            #           print(param[0])
      
        if learning_type== 'supervised':
            update_classifier_head(backbone, image_dim, num_classes, model_name = model_name )
        else:
            update_representation_head(backbone, image_dim, num_classes, model_name = model_name )
        self.backbone = backbone
        moving_average_decay = 0.01
        use_momentum = True
        self.use_momentum = use_momentum
        self.target_encoder = None
        self.student_ema_updater = EMA(moving_average_decay)

    def forward(self, images):
        
        classification_pred = self.backbone(images)

        return classification_pred
    
    

class SSLMODEL(nn.Module):
    def __init__(self, ssl_model, num_classes, image_dim = (3,224,224)):
        super(SSLMODEL, self).__init__()
        
        
        

        channel, height, width = image_dim
        
       
                 
        freeze_all_layers(ssl_model)    
                 
        # get shape after backbone 
        shape_size = get_model_output_shape(ssl_model, image_dim)
        
        # print backbone summary
        debug = False
        if debug:    
            summary(ssl_model, (channel, height, width))
            
            # for param in backbone.named_parameters():
            #       debug= True
            #       if debug:
            #           print(param[0])
      
        
        flatten_size = np.prod(list(shape_size))
        
        # set regression head
        hidden_size = int(flatten_size//4)
        
        if hidden_size < num_classes:
            assert False, 'needed to change classifier hidden size due amount of class'
        classifier_head = torch.nn.Sequential(
                                    nn.Dropout(p=0.25),
                                    nn.Linear(flatten_size, hidden_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.25),
                                    nn.Linear(hidden_size, num_classes)
                                    )
        self.ssl_model = ssl_model
        self.classifier_head = classifier_head
        

    def forward(self, images):
        
        ssl_representation = self.ssl_model(images)
        classification_pred = self.classifier_head(ssl_representation)

        return classification_pred
   







