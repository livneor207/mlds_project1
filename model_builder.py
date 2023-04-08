import torchvision.models as models
import torch.nn as nn
from torchsummary import summary
import torch 
import numpy as np




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



def update_classifier_head(backbone, image_dim, num_classes, model_name = 'efficientnet' ):
    # get shape after backbone 
    shape_size = get_output_shape(backbone, image_dim)
    flatten_size = np.prod(list(shape_size))
    
    # set regression head
    hidden_size = int(flatten_size*0.75)
    
    if hidden_size//4 < num_classes:
        assert False, 'needed to change classifier hidden size due amount of class'
   
    if model_name.find('resnet') != -1:
        # set classification head 
        backbone.fc = torch.nn.Sequential(
                                                    nn.Dropout(p=0.3),
                                                    nn.Linear(flatten_size, hidden_size),
                                                    nn.ReLU(inplace=True),
                                                    nn.Dropout(p=0.25),
                                                    nn.Linear(hidden_size, hidden_size//4),
                                                    nn.ReLU(inplace=True),
                                                    nn.Dropout(p=0.25),
                                                    nn.Linear(hidden_size//4,num_classes)
                                                    )
    elif model_name.find('efficientnet') != -1:
        # set classification head 
        backbone.classifier = torch.nn.Sequential(
                                                    nn.Dropout(p=0.3),
                                                    nn.Linear(flatten_size, hidden_size),
                                                    nn.ReLU(inplace=True),
                                                    nn.Dropout(p=0.25),
                                                    nn.Linear(hidden_size, hidden_size//4),
                                                    nn.ReLU(inplace=True),
                                                    nn.Dropout(p=0.25),
                                                    nn.Linear(hidden_size//4,num_classes)
                                                    )
        
    

class CNN(nn.Module):
    def __init__(self, num_classes, image_dim = (3,224,224), model_name = 'efficientnet_v2_m'):
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
      
        update_classifier_head(backbone, image_dim, num_classes, model_name = model_name )
        
        self.backbone = backbone


    def forward(self, images):
        
        classification_pred = self.backbone(images)

        return classification_pred
   







