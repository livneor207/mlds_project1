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

import numpy as np
import random
import gc 
import numpy as np
import random
import itertools
import sys
from scipy.spatial.distance import pdist, squareform,cdist
# def generate_max_hamming_permutations(amount_of_perm=4, max_allowed_perm=1000):
#     permutations = np.zeros((max_allowed_perm, amount_of_perm + 1), dtype=np.int32)
#     current_permutation = np.arange(amount_of_perm)

#     permutations[0, :-1] = current_permutation
#     i = 1

#     while i < max_allowed_perm and amount_of_perm > 1:
#         distances = np.count_nonzero(permutations[:i, :-1] != current_permutation, axis=1)
#         max_distance_index = np.argmax(distances)

#         available_indices = np.concatenate((current_permutation[:max_distance_index],
#                                              current_permutation[max_distance_index+1:]))
#         amount_of_perm -= 1

#         current_permutation = np.random.permutation(available_indices)
#         permutations[i, :-1] = current_permutation
#         i += 1

#     return permutations[:i, :-1]


def generate_max_hamming_permutations(amount_of_perm = 4, max_allowed_perm = 1000, amount_of_perm_to_generate = 100):
    """
      1. distances
        ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, 
        ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’,
        ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, 
        ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.
    2. amount of perm to generate - 
    """
    # all_permutation_option = list(permutations(range(0, amount_of_perm)))
    # permutations = np.array(list(itertools.permutations(range(amount_of_perm))))
    # random.shuffle(permutations)
    
    
    amount_of_permutation =  math.factorial(amount_of_perm)
    if max_allowed_perm>= amount_of_permutation:
        max_distance_permutations = np.array(list(permutations(range(0, amount_of_perm))))
        # np.random.shuffle(max_distance_permutations)

    else:
        # amount_of_perm_to_generate = 10
        # Generate the first permutation
        
        # generate first fermutation
        single_perm = np.arange(amount_of_perm)
        
    
        # current_permutation_index = random.randint(0, permutations.shape[0])
        single_perm = np.expand_dims(single_perm, axis = 0 )
        # permutations = np.delete(permutations, current_permutation_index, axis=0)
        max_distance_permutations =  np.zeros((max_allowed_perm, amount_of_perm))
        max_distance_permutations[0,:] = single_perm
        perm_index_array  =  np.zeros((max_allowed_perm, 1))

        i = 1
        while i<max_allowed_perm:
            if i>max_allowed_perm :
                break
            
            # generate n permutation
            sample_permutations = np.array([np.random.permutation(single_perm[0,:]) for _ in range(amount_of_perm_to_generate)])
            
            # Compute Hamming distances
            distances = cdist(sample_permutations,single_perm, metric='hamming')
           
            current_permutation_index = random.choice(np.where(distances == np.max(distances))[0])
            
            single_perm = np.expand_dims(sample_permutations[current_permutation_index, :], axis = 0 )
            
            t1 = time.time()
            perm_index = calculate_permutation_position(sample_permutations[current_permutation_index, :])
            t2 = time.time()-t1
            # permutations = np.delete(permutations, current_permutation_index, axis=0)
            row_exists = np.any(np.all(max_distance_permutations == single_perm, axis=1))
    
            if not row_exists:
               max_distance_permutations[i,:] = single_perm
               perm_index_array[i,0] = perm_index
               i+=1
            else:
               a=5
    
        max_distance_permutations = max_distance_permutations[0:i, :]
        perm_index_array = perm_index_array[0:i, :]
        sorted_indices = np.lexsort(max_distance_permutations.T[::-1])
        max_distance_permutations = np.int32(max_distance_permutations[sorted_indices])
        
    perm_index_list = []
    for i_perm in max_distance_permutations:
        perm_index = calculate_permutation_position(i_perm)
        perm_index_list.append(perm_index)
    
    max_distance_permutations = zip(max_distance_permutations, perm_index_list)
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
    elif task_name == 'CIFAR100':
         
        data_set = torchvision.datasets.CIFAR100('./', train=train, download=True)
        # trainval
        class_name_df =  pd.DataFrame(data_set.classes, columns = ['class_name'])
        class_name_df['class_index'] = np.arange(class_name_df.shape[0])
        data_df = pd.DataFrame(data_set.targets, columns = ['class_index'])
        data_df = pd.merge(data_df,class_name_df,  how = 'left', on = ['class_index'])
       
        data = data_set.data
    elif task_name == 'FOOD101':
        
        data_folder =  os.path.join(current_folder, 'food-101')


        data_set = torchvision.datasets.Food101(current_folder, download= True)
        
        class_dict = data_set.class_to_idx
        
        # trainval
        data_class_df =  pd.DataFrame(data_set.class_to_idx.items(), columns = ['class_name', 'class_index'])
        
        
        folder_path  = os.path.join(data_folder, 'images')
        
        class_folder_list = os.listdir(folder_path)
        
        data_df = pd.DataFrame()
        for i_folder in class_folder_list:
            path_2_find = os.path.join(folder_path, i_folder, '*.jpg' )
            images_path_list  = glob.glob(path_2_find)
            class_idx = class_dict[i_folder]
            
            temp_class_df = pd.DataFrame(images_path_list, columns = ['image_path'])
            temp_class_df['class_name'] = i_folder
            temp_class_df['class_index'] = class_idx
            
            data_df  = pd.concat([data_df, temp_class_df])

        data = None
        data_df = data_df.sample(frac=1).reset_index(drop=True)
        
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
               orig_pe = True, pe_dim = 128):
      
   
    self.pe_dim = pe_dim    
    self.all_permutation_option = copy.deepcopy(all_permutation_option)
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
    self.max_sentence_lenth = max(10000, 10*math.factorial(amount_of_patch))
    

    
    
    # self.max_sentence_lenth = math.factorial(amount_of_patch)
    if index_list is None:
        index_list= np.arange(0, data_df.shape[0])
    self.index_list = index_list
    
    amount_of_sampels = index_list.size

    self.pill_transform = transforms.ToPILImage()
    # r,z = np.unique(np.array(self.perm_order_list), return_counts= True, axis =0)
    # r,z2 = np.unique(np.array(self.perm_order_list2), return_counts= True, axis =0)
    self.all_permutation_option = list(self.all_permutation_option).copy()
    all_permutation_option_list, all_permutation_perm_index_list  = zip(*self.all_permutation_option)
    all_permutation_option_list = np.array(all_permutation_option_list).tolist()
    all_permutation_perm_index_list = np.array(all_permutation_perm_index_list).tolist()
    self.all_permutation_perm_index_list = all_permutation_perm_index_list
    self.all_permutation_option_list = all_permutation_option_list

    amount_of_perm = len(all_permutation_option_list)
    frequency =  amount_of_sampels//amount_of_perm+1
    k = amount_of_perm * frequency
    weighted_list = [element for element in self.all_permutation_option for _ in range(frequency)]
    self.perm_order_list = random.sample(weighted_list, amount_of_sampels)
    self.perm_order_list2 = random.sample(weighted_list, amount_of_sampels)
    a=5
    # self.perm_order_list = [random.choice(self.all_permutation_option) for _ in range(amount_of_sampels)]
    # self.perm_order_list2 = [random.choice(self.all_permutation_option) for _ in range(amount_of_sampels)]
    

    
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
    
    self.transform_test = transform_test
    self.color_transform = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.5)
    self.tranform_totensor  = transforms.ToTensor()
    self.transform_normelize = transforms.Normalize(means, stds)
    
    self.inverse_normalize = transforms.Compose([
                                                    transforms.Normalize(mean=[-m/s for m, s in zip(means, stds)],
                                                                         std=[1/s for s in stds])
                                                ])
    
    
    self.data_name = data_name
    self.amount_of_class = class_df.shape[0]
    self.class_df =  class_df
    self.class_list = self.class_df['class_name'].to_list()
    self.data_df  = data_df
    self.image_file_type = '.jpg'
    
    
  def __len__(self):
    return self.index_list.size
  
    
 
  def getPositionEncoding(self, perm_order, d, n=10000, perm_index = -1):
       # k = self.all_permutation_option.index(tuple(perm_order))
       perm_order = perm_order.tolist()
       if perm_index == -1:
           k2 = calculate_permutation_position(tuple(perm_order))
       else:
           k2 = perm_index
       k = self.all_permutation_option_list.index(perm_order)
       # k2 = k
       # amount_of_perm=  math.factorial(d)
       perm_label =  np.zeros((1,self.all_permutation_option.__len__()))

       perm_label[0,k]  = 1
       
       P = np.zeros((1, d))
       if not self.orig_pe:
           P[0,:] = perm_order
       else:
           for i in np.arange(int(d/2)):
              denominator = np.power(n, 2*i/d)
              P[0, 2*i] = np.sin(k2/denominator)
              if self.orig_pe:
                  P[0, 2*i+1] = np.cos(k2/denominator)
              else:
                  P[0, 2*i+1] = np.sin(k2/denominator)
       return P, perm_label
   
    
  def permutatation_aug(self, image):
     
      transform_image =  self.transform(image)
      # transform_image = image
      

      amount_of_patch = self.amount_of_patch
      dim_size = transform_image.shape[0]
      
      
      amount_of_rows = int(amount_of_patch**0.5)
      is_devided =  dim_size%amount_of_rows != 0

      matrix_order = gen_matrix_order(amount_of_rows)

      patch_row_size, patch_col_size = transform_image.shape[1]//amount_of_rows, transform_image.shape[2]//amount_of_rows
      # patch_array = patchify(transform_image.numpy(), (dim_size, patch_row_size, patch_col_size), patch_row_size)
      # # np_transform_image= transform_image.numpy()
      # if self.train:
      #      perm_order  = random.choice(self.all_permutation_option)
      # else:
      # random.shuffle(self.perm_order_list)
      # random.shuffle(self.perm_order_list2)


      if self.image_idx == 1:
            perm_order, perm_index = self.perm_order_list[self.idx]
      else:
            perm_order, perm_index = self.perm_order_list2[self.idx]
            if np.array_equal(perm_order, self.perm_order_list[self.idx][0]):
                perm_order, perm_index  = random.choice(self.perm_order_list2)

      # transform_image = self.inverse_normalize(transform_image)
      new_image = torch.zeros_like(transform_image)
      row = 0
      col = 0
      for index, i_perm in enumerate(perm_order):
          i_permutation_row = i_perm//amount_of_rows
          i_permutation_col = i_perm%amount_of_rows
          
          row = index//amount_of_rows
          col = index%amount_of_rows
          
          from_row = row*patch_row_size
          to_row = (row+1)*patch_row_size
          from_col = col*patch_col_size
          to_col =  (col+1)*patch_col_size
          
          # perm_order[index] = matrix_order[i_permutation_row, i_permutation_col]
          try:
          
              # patch_image = patch_array[0][i_permutation_row, i_permutation_col]
              from_row_perm = i_permutation_row*patch_row_size
              to_row_perm = (i_permutation_row+1)*patch_row_size
              from_col_perm = i_permutation_col*patch_col_size
              to_col_perm =  (i_permutation_col+1)*patch_col_size
              patch_image = transform_image[:,from_row_perm:to_row_perm,from_col_perm:to_col_perm] 
                            
          except:
              a=5 
          border_size = 0
          row_size, col_size = patch_image.shape[1::]
          # masked_patch = patch_image.copy()
          # masked_patch = patch_image.clone()
          
          # patch_image = self.color_transform(patch_image)
          
          # min_vals = image_tensor.min(dim=(2,3))[0]
          # max_vals = image_tensor.max(dim=(2, 3))[0]
          # t_norm = transforms.Normalize(self.means, self.stds)
          # patch_image = t_norm(patch_image)
          # padd_val = (np.array([[self.means]])*np.array([[self.stds]])).transpose(2,0,1)
         
          if index == 0:   
              # padd_val = random.choice([transform_image.max().item(),transform_image.min().item(),0,2.65,-2.17])
              # padd_val = random.choice([transform_image.max().item(),transform_image.min().item(),0.24, 2.65,-2.17])
              padd_val = random.choice([0])
              # padd_val = 2.67
              new_image += padd_val
          if row ==0 :
             patch_image[:,0:border_size*2,:]  = padd_val
          else:
              patch_image[:,0:border_size*2,:]  = padd_val
              
          if col == 0:
             # cols
             patch_image[:,:,0:border_size]  = padd_val
          else:
              patch_image[:,:,0:border_size]  =padd_val
          
          if  col == amount_of_rows-1 and is_devided:  
              patch_image[:,:,col_size-border_size::]  =  padd_val
          else:
              patch_image[:,:,col_size-border_size::]  =  padd_val
              
          if  row == amount_of_rows-1 and is_devided:  
              patch_image[:,row_size-border_size*2::,:]  = padd_val
          else:
              patch_image[:,row_size-border_size*2::,:]  = padd_val
          # patch_image2 = cv2.resize(cv2.copyMakeBorder(patch_image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, None, value = 0), dsize = patch_image.shape[1::], interpolation = cv2.INTER_AREA) 
          
          # new_image[0:dim_size, from_row:to_row, from_col:to_col] = torch.Tensor(masked_patch)
          new_image[0:dim_size, from_row:to_row, from_col:to_col] = patch_image

      perm_order, perm_label = self.getPositionEncoding(perm_order, self.pe_dim, n=self.max_sentence_lenth, perm_index = perm_index)
      # new_image = self.pill_transform(new_image)
      # new_image = self.tranform_totensor(new_image)
      # new_image = self.transform_normelize(new_image)
      
      return new_image, perm_order, perm_label
  
  def get_permutation_image(self, image):
      #
      if self.taske_name == 'perm' :
          start1 = time.time()
          if self.image_idx == 1 or 1:
              new_image, perm_order, perm_label = self.permutatation_aug(image)
          else:
              transform_image =  self.transform(image)
              # transform_image = self.inverse_normalize(transform_image)
              # transform_image = self.color_transform(transform_image)
              # transform_image = self.pill_transform(transform_image)
              # transform_image = self.tranform_totensor(transform_image)
              # transform_image = self.transform_normelize(transform_image)
              # transform_image =  image
              perm_order = torch.empty((1,self.pe_dim))
              new_image = transform_image
              perm_label = np.zeros((1,len(self.all_permutation_option_list)))
              
          count_time = time.time()-start1
          # print(count_time)
          a=5 
      else:
          # transform_image = self.transform_test(image)
          transform_image =  self.transform(image)
         
          # transform_image =  image

          perm_order = torch.empty((1,self.pe_dim))
          new_image = transform_image
          perm_label = np.zeros((1,len(self.all_permutation_option_list)))
      perm_order = torch.Tensor(perm_order)
      perm_label = torch.Tensor(perm_label)

      return new_image, perm_order, perm_label
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
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            amount_of_tries = 4
            tries_index = 1
            while  tries_index < amount_of_tries:
                try:
                    data_df_row= self.data_df.sample().squeeze(axis=0)
                    image_path = data_df_row['image_path']
                    label_file_name = data_df_row['class_index']
                    image = Image.open(image_path).convert('RGB')
                    break
                except:
                    pass
                tries_index += 1

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
    # transform_image =  self.transform(image)

    new_image, perm_order,  perm_label = self.get_permutation_image(image)
    total =  t1 - time.time()
    self.image_idx += 1
    if desire_amount_of_images > 1:
        # transform_image =  self.transform(image)

        new_image2, prem_order2, perm_label2= self.get_permutation_image(image)
        new_image = torch.concatenate([new_image, new_image2])
        perm_order = torch.concatenate([perm_order, prem_order2])
        perm_label = torch.concatenate([perm_label, perm_label2])

        
    transform_image = new_image
    # set sample
    sample = (transform_image, label, torch.Tensor(perm_order), label_name, perm_label)
    
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
    max_allowed_permutation = training_configuration.max_allowed_permutation
    pe_dim = training_configuration.pe_dim

    
    # all_permutation_option = list(permutations(range(0, amount_of_patch)))
    # all_permutation_option = [] 
    # all_permutation_option = generate_max_hamming_permutations(amount_of_perm = amount_of_patch, max_allowed_perm = max_allowed_permutation, amount_of_perm_to_generate = 100)
    all_permutation_option = training_configuration.all_permutation_option
    # all_permutation_option = list(all_permutation_option).copy()
    # all_permutation_option_list, all_permutation_perm_index_list  = zip(*all_permutation_option)
    # all_permutation_option_list = np.array(all_permutation_option_list).tolist()
      
    #   return np.clip(output, clip_min, 1.)
    # # all_permutation_option = np.array(list(permutations(range(0, amount_of_patch))))
    # # 
    # # position_embeding = getPositionEncoding(seq_len=75, d=512, n=10000)
    # position_embeding = all_permutation_array

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
    
    
    # distances = cdist(position_embeding, position_embeding, metric='cosine')
    # distances2 = cdist(position_embeding, position_embeding, metric='euclidean')

    
    # permutation_dictionary = dict(zip(all_permutation_option, position_embeding))
    
    # np.linalg.norm(position_embeding[0] - position_embeding[100],1)
    # all_permutation_option[0]
    # all_permutation_option[20000]

    # permutation_dictionary[tuple(perm_order)]
    
    
    
    tasks_list = ['perm', 'no_perm']
    if not taske_name in  tasks_list:
        assert False, 'task not defined'
        
    if learning_type == 'supervised':
        p = 0.33
    else:
        p = 1
    center_crop_size = int(0.9*image_size)

    # resize_transforms = transforms.Resize((image_size,image_size), interpolation = transforms.InterpolationMode.NEAREST_EXACT)\
    resize_transforms = transforms.Resize((image_size,image_size), interpolation = transforms.InterpolationMode.BILINEAR  )
    # resize_transforms = transforms.Resize((image_size,image_size), interpolation = transforms.InterpolationMode.LANCZOS )
  
    transformations = []
    if rand_choise:
        
        if taske_name == 'perm':
            min_scale = 0.4
            transformations  = [
                                transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.1)], p=0.9),
                                transforms.RandomHorizontalFlip(p=0.5),
                                # torchvision.transforms.RandomVerticalFlip(p=0.5),
                                transforms.RandomResizedCrop(size = (image_size, image_size), scale=(min_scale, 1.0)),
                                transforms.RandomGrayscale(p=0.5),
                                transforms.RandomApply([transforms.GaussianBlur(kernel_size= 3, sigma = (0.1, 2))],p = 0.25)   
                                ]
        else:
            min_scale = 0.2
        
            transformations  = [transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                transforms.RandomHorizontalFlip(p=0.5),
                                # torchvision.transforms.RandomVerticalFlip(p=0.5),
                                transforms.RandomResizedCrop(size = (image_size, image_size), scale=(min_scale, 1.0)),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.RandomApply([transforms.GaussianBlur(kernel_size= 3, sigma = (0.1, 2))],p = 0.5)   
                                ]
            
            
        # transformations = [
        #                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5)
        #                     ]
        # transformations = [
        #     transforms.RandomApply(
        #     [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        #     transforms.RandomHorizontalFlip(),
        #     # transforms.RandomResizedCrop(size = (image_size, image_size), scale=(min_scale, 1.0)),
        #     transforms.RandomGrayscale(p=0.2)
        #     # transforms.RandomApply(
        #     #     [transforms.GaussianBlur(kernel_size= 3, sigma = (0.1, 2))],
        #     #     p = 0.5
        #     # )
        #     ]
        # transformations += additinal_aug
    
            
        # transformations = [     transforms.RandomHorizontalFlip(p=p),
        #                         transforms.ColorJitter(brightness=.25, hue=.25,saturation = 0.25, contrast = 0.25)]
        #                         # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        #                         # transforms.RandomCrop(size=(center_crop_size, center_crop_size))]
                            
        if not taske_name == 'perm':
            # transformations += [transforms.RandomCrop(size=(center_crop_size, center_crop_size)),
            #                     transforms.RandomHorizontalFlip(p=p)]
            pass
        rand_choise = transforms.RandomChoice(transformations)
    else:
        rand_choise = transforms.RandomChoice( [transforms.RandomHorizontalFlip(p=0)])
    if taske_name == 'perm':
        data_transforms =   transforms.Compose([rand_choise,
                                                resize_transforms,
                                                # resize_transforms,
                                                transforms.ToTensor(),
                                                transforms.Normalize(means, stds)
                                                ])
    elif taske_name == 'no_perm':
        data_transforms =   transforms.Compose([
                                                rand_choise,
                                                resize_transforms,
                                                # transforms.CenterCrop((center_crop_size,center_crop_size)),
                                                # resize_transforms,
                                                transforms.ToTensor(),
                                                transforms.Normalize(means, stds)
                                                ])
    
    test_transforms =  transforms.Compose([ resize_transforms,
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
                        all_permutation_option=all_permutation_option, orig_pe = orig_pe, pe_dim=pe_dim)
    X_val = MyDataset(all_train_df,class_df, data_transforms, test_transforms ,index_list = val_index, amount_of_patch=amount_of_patch, 
                      train = False, data_name = 'val', taske_name = taske_name, learning_type = learning_type, data=train_data, 
                      all_permutation_option=all_permutation_option, orig_pe = orig_pe, pe_dim=pe_dim)
    X_test = MyDataset(test_df, class_df, data_transforms, test_transforms , index_list = None, amount_of_patch=amount_of_patch,
                       train = False, data_name = 'test', taske_name = taske_name, learning_type = learning_type, data=test_data, 
                       all_permutation_option=all_permutation_option, orig_pe = orig_pe, pe_dim=pe_dim)

       
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size,shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = torch.utils.data.DataLoader(X_val, batch_size=batch_size,shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    debug_loader = torch.utils.data.DataLoader(copy.deepcopy(X_train), batch_size=debug_batch_size, shuffle=True, pin_memory=False)
    # if not tb_writer is None and 0:
    #     add_data_embedings(val_loader, tb_writer, n=300)

    return train_loader, val_loader, test_loader, debug_loader