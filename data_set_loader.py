import os 
import random
import numpy as np
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Subset
from torch.utils.data import Dataset
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import copy
from tensorboard_hellper import add_data_embedings
import pandas as pd
import patchify
from patchify import patchify
from utils import *
import torchvision
import time 
import itertools
from itertools import permutations
import math


def generate_max_hamming_permutations(amount_of_perm = 4, max_allowed_perm = 1000):
    """
     1. distances
        ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, 
        ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’,
        ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, 
        ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.
    2. amount of perm to generate - 
    """
    permutations = np.array(list(itertools.permutations(range(amount_of_perm))))
    # random.shuffle(permutations)
    current_perm_index = random.randint(0, permutations.shape[0])
    single_perm = np.expand_dims(permutations[current_perm_index, :], axis = 0 )
    permutations = np.delete(permutations, current_perm_index, axis=0)
    max_distance_permutations =  np.zeros((max_allowed_perm, permutations.shape[1]))
    max_distance_permutations[0,:] = single_perm
    i = 1
    while i<max_allowed_perm:
        if i>max_allowed_perm or permutations.shape[0] == 0:
            break
        # Compute Hamming distances
        distances = cdist(permutations, single_perm, metric='hamming')
       
        current_perm_index = random.choice(np.where(distances == np.max(distances))[0])
        single_perm = np.expand_dims(permutations[current_perm_index, :], axis = 0 )

        permutations = np.delete(permutations, current_perm_index, axis=0)
        max_distance_permutations[i,:] = single_perm

        i+=1
    max_distance_permutations = max_distance_permutations[0:i, :]
    return max_distance_permutations


def get_statistic_from_stistic_dataframe(train_statistic_df):
    class_ratios = train_statistic_df['alpha'].to_numpy()
    amount_of_class = train_statistic_df.shape[0] 
    return class_ratios, amount_of_class

def data_statistics(train_df):
    counts_df = train_df[['class_name']].copy()
    counts_df =  counts_df.drop_duplicates(subset=['class_name'])
    unisque_index , counts  = np.unique(train_df['class_name'], return_counts= True)
    alpha = np.round(counts/np.sum(counts),2)
    temp_df = pd.DataFrame()
    temp_df['alpha'] = alpha
    temp_df['class_name'] = unisque_index 
    train_statistic_df = pd.merge(temp_df, counts_df , how = 'right', on = ['class_name'] )
    return train_statistic_df, alpha

def change_file_ending(file_path, new_file_extension ):
    # Remove the old file extension
    file_name = os.path.splitext(file_path)[0]
    
    
    # Add the new file extension to the file name
    new_file_path = file_name + new_file_extension
    
    return new_file_path


def parse_train_data(task_name = 'cat_dogs', folder_path = '', train = True, current_folder = ''):
    if task_name == 'cat_dogs':
        
        # get all files names 
        images_path_list = get_all_images_from_specific_folder(folder_path)
        
        class_name_list = list(map(lambda x: os.path.basename(x).split('.')[0], images_path_list))
        data_class_df = pd.DataFrame([['cat', 0],['dog',1]], columns = ['class_name', 'class_index'])
        data_df = pd.DataFrame(class_name_list, columns = ['class_name'])
        data_df['image_path'] = images_path_list
        data_df = pd.merge(data_df, data_class_df, how = 'left', on = ['class_name'])
        if data_df.isnull().values.any():
            data_df = data_df[['image_path']]
        data = None
        data_df = data_df.sample(frac=1).reset_index(drop=True)

    elif task_name == 'CIFAR10':
         
        data_set = torchvision.datasets.CIFAR10('./', train=train, download=True)
        # trainval
        class_name_df =  pd.DataFrame(data_set.classes, columns = ['class_name'])
        class_name_df['class_index'] = np.arange(class_name_df.shape[0])
        data_df = pd.DataFrame(data_set.targets, columns = ['class_index'])
        data_df = pd.merge(data_df,class_name_df,  how = 'left', on = ['class_index'])
       
        data = data_set.data
    elif task_name == 'OxfordIIITPet':
        
        data_folder =  os.path.join(current_folder, 'Pets')

        data_set = torchvision.datasets.OxfordIIITPet(root=data_folder, split  = train, download=True)
        
        data_class_df =  pd.DataFrame(data_set.class_to_idx.items(), columns = ['class_name', 'class_index'])
        folder_path  = os.path.join(data_folder, 'oxford-iiit-pet', 'images')
        # get all files names 
        images_path_list = get_all_images_from_specific_folder(folder_path)
        
        class_name_list = list(map(lambda x: (' ').join(os.path.basename(x).split('_')[0:-1]), images_path_list))
        data_df = pd.DataFrame(class_name_list, columns = ['class_name'])
        data_df['image_path'] = images_path_list
        
        data_class_df['class_name'] = data_class_df['class_name'].str.lower()
        data_df['class_name'] = data_df['class_name'].str.lower()



        data_df = pd.merge(data_df, data_class_df, how = 'left', on = ['class_name'])
        if data_df.isnull().values.any():
            data_df = data_df[['image_path']]
        data = None
        data_df = data_df.sample(frac=1).reset_index(drop=True)
    
        
    return data_df, data


def parse_test_data(images_path_list):
    test_df = pd.DataFrame()
    test_df['image_path'] = images_path_list
    return test_df

def gen_matrix_order(n):
     # n = 4
     grid = np.zeros((n, n))
     
     for i in range(n):
         for j in range(n):
             distances = []
             for k in range(n):
                 for l in range(n):
                     distance = np.sqrt((k - i)**2 + (l - j)**2)
                     distances.append(distance)
             distances.remove(0)
             weight = sum([1/d for d in distances])
             grid[i, j] = weight + (i% n+  n*j)*0.1
     # matrix_order = grid/grid.max()
     matrix_order = grid
     # print(matrix_order)
     return matrix_order
def getPositionEncoding(seq_len, d, n=10000):
      P = np.zeros((seq_len, d))
      for k in range(seq_len):
          for i in np.arange(int(d/2)):
              denominator = np.power(n, 2*i/d)
              P[k, 2*i] = np.sin(k/denominator)
              P[k, 2*i+1] = np.cos(k/denominator)
          torch.Tensor(P[k])
      return P
 
    

def calculate_permutation_position(permutation):
    n = len(permutation)
    position = 0
    for i, num in enumerate(permutation):
        count = sum(num > p for p in permutation[i+1:])
        position += count * math.factorial(n-1-i)
    return position 

class MyDataset(Dataset):

  def __init__(self, data_df,  
               class_df,  
               transform_train,
               transform_test,
               index_list = None, 
               amount_of_patch = 4, 
               train = True, 
               data_name = 'train',
               debug = False, 
               max_debug_image_allowed = 0, 
               means = [0.485, 0.456, 0.406], 
               stds=[0.229, 0.224, 0.225], 
               taske_name = 'perm' , 
               learning_type = 'supervised',
               data=None,
               all_permutation_option= None,
               orig_pe = True):
    
    self.all_permutation_option = all_permutation_option
    self.taske_name = taske_name
    self.amount_of_patch = amount_of_patch
    self.means = means
    self.stds = stds
    self.debug = debug
    self.debug_image_idx = 0
    self.max_debug_image_allowed = max_debug_image_allowed
    self.train = train
    self.learning_type = learning_type
    self.orig_pe = orig_pe
    amount_of_sampels = data_df.shape[0]
    if index_list is None:
        index_list= np.arange(0, data_df.shape[0])
    self.index_list = index_list
    self.pill_transform = transforms.ToPILImage()
    self.perm_order_list = [random.sample(range(amount_of_patch), amount_of_patch) for _ in range(amount_of_sampels)]
    self.perm_order_list2 = [random.sample(range(amount_of_patch), amount_of_patch) for _ in range(amount_of_sampels)]

    if data is None:
        read_image = True
    else:
        read_image = False
    self.read_image= read_image
    self.data =  data
    if train :
        self.transform = transform_train
    else:
        self.transform = transform_test
    
    self.data_name = data_name
    self.amount_of_class = class_df.shape[0]
    self.class_df =  class_df
    self.class_list = self.class_df['class_name'].to_list()
    self.data_df  = data_df
    self.image_file_type = '.jpg'
    
    
  def __len__(self):
    return self.index_list.size
  
    
 
  def getPositionEncoding(self, perm_order, d, n=10000):
       # k = self.all_permutation_option.index(tuple(perm_order))
       k = calculate_permutation_position(tuple(perm_order))
     
       # amount_of_perm=  math.factorial(d)
       
       P = np.zeros((1, d))
       for i in np.arange(int(d/2)):
          denominator = np.power(n, 2*i/d)
          P[0, 2*i] = np.sin(k/denominator)
          if self.orig_pe:
              P[0, 2*i+1] = np.cos(k/denominator)
          else:
              P[0, 2*i+1] = np.sin(k/denominator)
       return P
   
    
  def permutatation_aug(self, image):
      
      
      
     
      transform_image =  self.transform(image)

      

      amount_of_patch = self.amount_of_patch
      dim_size = transform_image.shape[0]
      amount_of_rows = int(amount_of_patch**0.5)
      matrix_order = gen_matrix_order(amount_of_rows)

      patch_row_size, patch_col_size = transform_image.shape[1]//amount_of_rows, transform_image.shape[2]//amount_of_rows
      patch_array = patchify(transform_image.numpy(), (dim_size, patch_row_size, patch_col_size), patch_row_size)
      if self.train:
           perm_order  = random.sample(range(amount_of_patch), amount_of_patch)
      else:
          if self.image_idx == 1:
              perm_order = self.perm_order_list[self.idx]
          else:
              perm_order = self.perm_order_list2[self.idx]

      
      new_image = torch.zeros_like(transform_image)
      row = 0
      col = 0
      for index, i_perm in enumerate(perm_order):
          i_perm_row = i_perm//amount_of_rows
          i_perm_col = i_perm%amount_of_rows
          
          row = index//amount_of_rows
          col = index%amount_of_rows
          
          from_row = row*patch_row_size
          to_row = (row+1)*patch_row_size
          from_col = col*patch_col_size
          to_col =  (col+1)*patch_col_size
          
          # perm_order[index] = matrix_order[i_perm_row, i_perm_col]

          
          patch_image = patch_array[0][i_perm_row, i_perm_col]
          border_size = 0
          row_size, col_size = patch_image.shape[1::]
          masked_patch = patch_image.copy()
          # padd_val = (np.array([[self.means]])*np.array([[self.stds]])).transpose(2,0,1)
          padd_val = 0
          masked_patch[:,:,0:border_size]  = padd_val
          masked_patch[:,:,col_size-border_size::]  =  padd_val
          masked_patch[:,0:border_size,:]  = padd_val
          masked_patch[:,row_size-border_size::,:]  = padd_val
          # patch_image2 = cv2.resize(cv2.copyMakeBorder(patch_image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, None, value = 0), dsize = patch_image.shape[1::], interpolation = cv2.INTER_AREA) 
          
          
          new_image[0:dim_size, from_row:to_row, from_col:to_col] = torch.Tensor(masked_patch)
     
      perm_order = self.getPositionEncoding(perm_order, amount_of_patch, n=10000)
      
      return new_image, perm_order
  
  def get_perm_image(self, image):
      if self.taske_name == 'perm':
          
          new_image, perm_order = self.permutatation_aug(image)
      else:
          transform_image =  self.transform(image)

          perm_order = torch.empty(self.amount_of_patch)
          new_image = transform_image
      perm_order = torch.Tensor(perm_order)
      return new_image, perm_order
  def generate_original_image_plot(self, image, axarr):
      np_image_0_1 =  np.array(image)
      np_image_0_1 = np_image_0_1.astype(np.uint8)
      axarr[0].imshow(np_image_0_1)
      axarr[0].title.set_text('image before transformation')
  def generate_transformed_image_plot(self, np_transform_image_0_1, axarr, image_index = 1):
      np_transform_image_0_1 *= np.array(self.stds)
      np_transform_image_0_1 += np.array(self.means)
      np_transform_image_0_1*=255
      np_transform_image_0_1 = np_transform_image_0_1.astype(np.uint8)
      axarr[image_index].imshow(np_transform_image_0_1)
      axarr[image_index].title.set_text(f'image after transformation {image_index}')
      
    
  def __getitem__(self, idx):
    # get image
    
    data_df_row = self.data_df.iloc[self.index_list[idx]]
    self.idx = idx
    label_file_name = data_df_row['class_index']
    if self.read_image:
        image_path = data_df_row['image_path']
        image = Image.open(image_path).convert('RGB')
        
    else:
        image = self.data[self.index_list[idx]]
        image = self.pill_transform(image)
    if self.data_name in ['train', 'val']:
        label_target = data_df_row['class_index']
        label_name = data_df_row['class_name']
    else:
        label_target = -1
        label_name = 'unknon'
        
    label =  np.zeros(( self.amount_of_class))
    label[label_target] = 1
    label =  torch.Tensor(label)
    label = label.to(torch.float)
    
    # transform_image =  self.transform(image)
    
    if self.learning_type == 'supervised':
        desire_amount_of_images = 1
    else:
        desire_amount_of_images = 2
    t1 = time.time()
    self.image_idx = 1
    new_image, perm_order = self.get_perm_image(image)
    total =  t1 - time.time()
    self.image_idx += 1
    if desire_amount_of_images > 1:
        # transform_image =  self.transform(image)

        new_image2, prem_order2 = self.get_perm_image(image)
        new_image = torch.concatenate([new_image, new_image2])
        perm_order = torch.concatenate([perm_order, prem_order2])

        
    transform_image = new_image
    # set sample
    sample = (transform_image, label, torch.Tensor(perm_order), label_name)
    
    if self.debug:
      
        np_transform_image_0_1 =  transform_image.numpy()
        np_transform_image_0_1 = np_transform_image_0_1.transpose(1,2,0)
        
        if desire_amount_of_images<=1:
            f, axarr = plt.subplots(1,2, figsize=(10,20))
        else:
            f, axarr = plt.subplots(1,3, figsize=(10,30))
        self.generate_original_image_plot(image, axarr)
        self.generate_transformed_image_plot(np_transform_image_0_1[:,:,0:3], axarr, image_index = 1)
        if desire_amount_of_images>1:
            self.generate_transformed_image_plot(np_transform_image_0_1[:,:,3::], axarr, image_index = 2)

        
      
            
        f.suptitle(f'Class is {label_name}') # or plt.suptitle('Main title')


    return sample


def initialize_dataloaders(all_train_df,  test_df, training_configuration, amount_of_patch = 4 ,batch_size=8, val_split=0.1, debug_batch_size=8, random_state=1001,
                           means = [0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225], image_size = 224, tb_writer = None, taske_name = 'perm',
                           learning_type = 'supervised', num_workers = 2, train_data = None, test_data = None, rand_choise = True,
                           pin_memory=False, orig_pe = True, train_split = 1):
    
    batch_size = training_configuration.batch_size
    amount_of_patch = training_configuration.amount_of_patch
    taske_name = training_configuration.perm
    learning_type = training_configuration.learning_type
    num_workers = training_configuration.num_workers
    
    
    
    # all_permutation_option = list(permutations(range(0, amount_of_patch)))
    all_permutation_option = [] 
    
    # def cosine_schedule(k=0, d=4):
      
    #   amount_of_perm=  math.factorial(d)
    #   all_perm_array = np.zeros((amount_of_perm,d))

    #   for k in range(amount_of_perm):
    #       empty_array = np.zeros((1,d))
    #       for i in range(d):
    #           nominator =  (i/d)+k
    #           denominator = 1+k
    #           fraction = nominator/denominator
    #           # fraction *= (np.pi/2)
    #           val = math.cos(fraction) ** (0.5)
    #           empty_array[0, i] = val
    #       all_perm_array[k, :] = empty_array
          
      
    #   return np.clip(output, clip_min, 1.)
    # # all_permutation_option = np.array(list(permutations(range(0, amount_of_patch))))
    # # 
    # # position_embeding = getPositionEncoding(seq_len=len(all_permutation_option), d=24, n=10e3)
    # position_embeding = all_perm_array

    # Create a sample NumPy array
   
    # def position_encoding(max_length, d_model):
    #     positions = np.arange(max_length)[:, np.newaxis]
    #     indices = np.arange(d_model)[np.newaxis, :]
    #     angles = positions / np.power(10000, 2 * (indices // 2) / d_model)
    #     sin_values = np.sin(angles[:, 0::2])
    #     cos_values = np.cos(angles[:, 1::2])
    #     position_encodings = np.concatenate([sin_values, cos_values], axis=-1)
    #     return position_encodings
    
   
    
    

    # position_embeding = position_encoding(24, 24)
    # position_embeding =  position_embeding[:,0:4]
    # # Calculate pairwise KLD between rows
    # from scipy.special import kl_div
    # softmax_array = np.exp(position_embeding) / np.exp(position_embeding).sum(axis=1, keepdims=True)

    # klds = np.zeros((softmax_array.shape[0], softmax_array.shape[0]))
    
    # for i in range(softmax_array.shape[0]-1):
    #     kld = kl_div(softmax_array[i+1:], softmax_array[i]).sum(axis=1)
    #     klds[i, i+1:] = kld
    #     klds[i+1:, i] = kld
    
    # # Print the adjacency matrix
    # print(klds)
    
    
    # from scipy.spatial.distance import pdist, squareform,cdist

    # # # # Calculate pairwise distances between rows
    # distances = pdist(position_embeding)
    
    # # # #Convert the condensed distance matrix to a square matrix
    # adjacency_matrix = squareform(distances)
    
    
    # distances = cdist(position_embeding, position_embeding, metric='cityblock')


    
    # permutation_dictionary = dict(zip(all_permutation_option, position_embeding))
    
    # np.linalg.norm(position_embeding[0] - position_embeding[100],1)
    # all_permutation_option[0]
    # all_permutation_option[20000]

    # permutation_dictionary[tuple(perm_order)]
    
    
    
    tasks_list = ['perm', 'no_perm']
    if not taske_name in  tasks_list:
        assert False, 'task not defined'
        
    if learning_type == 'supervised':
        p = 0.5
    else:
        p = 1
    center_crop_size = int(0.9*image_size)

    resize_transforms = transforms.Resize((image_size,image_size), interpolation = transforms.InterpolationMode.NEAREST_EXACT)\
    # resize_transforms = transforms.Resize((image_size,image_size), interpolation = transforms.InterpolationMode.BILINEAR)
    if rand_choise:
        rand_choise = transforms.RandomChoice( [
                                  transforms.RandomCrop(size=(center_crop_size, center_crop_size)),
                                  transforms.RandomHorizontalFlip(p=p),
                                  transforms.ColorJitter(brightness=.25, hue=.25,saturation = 0.25, contrast = 0.25),
                                  transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))])
    else:
        rand_choise = transforms.RandomChoice( [transforms.RandomHorizontalFlip(p=0)])
    if taske_name == 'perm':
        data_transforms =   transforms.Compose([resize_transforms,
                                                rand_choise,
                                                resize_transforms,
                                                transforms.ToTensor(),
                                                transforms.Normalize(means, stds)])
    elif taske_name == 'no_perm':
        data_transforms =   transforms.Compose([resize_transforms,
                                                
                                                rand_choise,
                                                # transforms.CenterCrop((center_crop_size,center_crop_size)),
                                                resize_transforms,
                                                transforms.ToTensor(),
                                                transforms.Normalize(means, stds)])
    
    test_transforms =  transforms.Compose([ resize_transforms,
                                            # transforms.CenterCrop((center_crop_size, center_crop_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(means, stds)])

    # get amount of class
    class_df =  all_train_df[['class_name', 'class_index']]
    class_df =  class_df.drop_duplicates(subset=['class_name'])
    
    # targets =  train_dataset.targets
    try:
        train_index, val_index = train_test_split(np.arange(all_train_df.shape[0]), stratify = all_train_df['class_index'] ,  test_size=val_split, random_state=random_state)
    except:
        train_index, val_index = train_test_split(np.arange(all_train_df.shape[0]) ,  test_size=val_split, random_state=random_state)

    if train_split!=1:
        try:
            train_index, dummy = train_test_split(train_index, stratify = all_train_df['class_index'][train_index] ,  test_size=1-train_split, random_state=random_state)
        except:
            train_index, dummy = train_test_split(train_index ,  test_size=1-train_split, random_state=random_state)



    # train_df, val_df = train_test_split(all_train_df, stratify = all_train_df['class_name'],  test_size=val_split, random_state=random_state)

    
    X_train = MyDataset(all_train_df, class_df, data_transforms, test_transforms ,index_list = train_index, amount_of_patch=amount_of_patch, 
                        train = True, data_name = 'train', taske_name = taske_name, learning_type = learning_type, data=train_data,
                        all_permutation_option=all_permutation_option, orig_pe = orig_pe)
    X_val = MyDataset(all_train_df,class_df, data_transforms, test_transforms ,index_list = val_index, amount_of_patch=amount_of_patch, 
                      train = False, data_name = 'val', taske_name = taske_name, learning_type = learning_type, data=train_data, 
                      all_permutation_option=all_permutation_option, orig_pe = orig_pe)
    X_test = MyDataset(test_df, class_df, data_transforms, test_transforms , index_list = None, amount_of_patch=amount_of_patch,
                       train = False, data_name = 'test', taske_name = taske_name, learning_type = learning_type, data=test_data, 
                       all_permutation_option=all_permutation_option, orig_pe = orig_pe)

       
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size,shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = torch.utils.data.DataLoader(X_val, batch_size=batch_size,shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    debug_loader = torch.utils.data.DataLoader(copy.deepcopy(X_train), batch_size=debug_batch_size, shuffle=True, pin_memory=False)
    # if not tb_writer is None and 0:
    #     add_data_embedings(val_loader, tb_writer, n=300)

    return train_loader, val_loader, test_loader, debug_loader