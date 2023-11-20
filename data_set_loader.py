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


def generate_max_hamming_permutations(amount_of_perm = 4, max_allowed_perm = 1000, amount_of_perm_to_generate = 100):
    """
      1. distances
        ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, 
        ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’,
        ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, 
        ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.
    2. amount of perm to generate - 
    """
    """
    generate subgroup of pemutation base there distance from each other, the
    output is list in predefined size of permutation
    """
    
    # amount of permutation that can be generated 
    amount_of_permutation =  math.factorial(amount_of_perm)
    
    # if this number is greater the amount of posible permutation to generate generate all  
    if max_allowed_perm>= amount_of_permutation:
        max_distance_permutations = np.array(list(permutations(range(0, amount_of_perm))))

    else:

        # generate first fermutation
        single_perm = np.arange(amount_of_perm)
        
    
        # current_permutation_index = random.randint(0, permutations.shape[0])
        single_perm = np.expand_dims(single_perm, axis = 0 )
        # permutations = np.delete(permutations, current_permutation_index, axis=0)
        max_distance_permutations =  np.zeros((max_allowed_perm, amount_of_perm))
        max_distance_permutations[0,:] = single_perm
        perm_index_array  =  np.zeros((max_allowed_perm, 1))
        # each time add new permutation which is most similar to last permutation, with the 
        # goal that the semantic info will be not loss from last choosen permutation 
        i = 1
        while i<max_allowed_perm:
            if i>max_allowed_perm :
                break
            
            # generate n permutation
            sample_permutations = np.array([np.random.permutation(single_perm[0,:]) for _ in range(amount_of_perm_to_generate)])
            
            # Compute Hamming distances
            distances = cdist(sample_permutations,single_perm, metric='cosine')
           
            # choose next permutation base on min distance
            current_permutation_index = random.choice(np.where(distances == np.min(distances))[0])
            
            # expand dimentions of choosen permutatino 
            single_perm = np.expand_dims(sample_permutations[current_permutation_index, :], axis = 0 )
            
            t1 = time.time()
            # calculate permutation index 
            perm_index = calculate_permutation_position(sample_permutations[current_permutation_index, :])
            t2 = time.time()-t1
            
            # check if permutation was choosen before
            row_exists = np.any(np.all(max_distance_permutations == single_perm, axis=1))
            
            # add permutation 
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
    """
    get amount of class and thire ratios
    """
    class_ratios = train_statistic_df['alpha'].to_numpy()
    amount_of_class = train_statistic_df.shape[0] 
    return class_ratios, amount_of_class

def data_statistics(train_df):
    """
    generate statistic connect to the class balance 
    """
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
    """
    change file ending if needed 
    """
    
    # Remove the old file extension
    file_name = os.path.splitext(file_path)[0]
    
    
    # Add the new file extension to the file name
    new_file_path = file_name + new_file_extension
    
    return new_file_path


def parse_train_data(task_name = 'cat_dogs', folder_path = '', train = True, current_folder = ''):
    """
    

    Parameters
    ----------
    task_name : dataset name 
    folder_path : if download dataset where to puthim
    train : load train\test dataset
    current_folder : only for dog & cats data set the data is download before
    Returns
    -------
    data_df : data frame of file path, with class per row, and label index 
    data : data if exist (CIFAR)

    """
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
    """
    generate dataframe base list of file path 
    """
    
    test_df = pd.DataFrame()
    test_df['image_path'] = images_path_list
    return test_df

def getPositionEncoding(seq_len, d, n=10000):
      """
      generate postion embedding vec

      """
      P = np.zeros((seq_len, d))
      for k in range(seq_len):
          for i in np.arange(int(d/2)):
              denominator = np.power(n, 2*i/d)
              P[k, 2*i] = np.sin(k/denominator)
              P[k, 2*i+1] = np.cos(k/denominator)
          torch.Tensor(P[k])
      return P
 
    

def calculate_permutation_position(permutation):
    """
    Parameters
    ----------
    permutation : permutation vec 
    Returns
    -------
    position : for given permutation return the permutation index 
    """
    n = len(permutation)
    position = 0
    for i, num in enumerate(permutation):
        count = sum(num > p for p in permutation[i+1:])
        position += count * math.factorial(n-1-i)
    return position 

class MyDataset(Dataset):
  """
    generate custom dataset
  """

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
      
   
    
    # initiate dataset parameters 
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
    self.data_name = data_name
    self.amount_of_class = class_df.shape[0]
    self.class_df =  class_df
    self.class_list = self.class_df['class_name'].to_list()
    self.data_df  = data_df
    self.image_file_type = '.jpg'
    self.data =  data
        
    # self.max_sentence_lenth = math.factorial(amount_of_patch)
    if index_list is None:
        index_list= np.arange(0, data_df.shape[0])
        
    # set sample index list 
    self.index_list = index_list
    
    # amount of sample
    amount_of_sampels = index_list.size

    # manually transformation 
    self.pill_transform = transforms.ToPILImage()
    self.transform_test = transform_test
    self.color_transform = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.5)
    self.tranform_totensor  = transforms.ToTensor()
    self.transform_normelize = transforms.Normalize(means, stds)
    self.inverse_normalize = transforms.Compose([
                                                    transforms.Normalize(mean=[-m/s for m, s in zip(means, stds)],
                                                                         std=[1/s for s in stds])
                                                ])
    
    
    
    # get all permutation options
    self.all_permutation_option = list(self.all_permutation_option).copy()
    
    # generate all permutation list and all permutation concat with permutation index 
    all_permutation_option_list, all_permutation_perm_index_list  = zip(*self.all_permutation_option)
    all_permutation_option_list = np.array(all_permutation_option_list).tolist()
    all_permutation_perm_index_list = np.array(all_permutation_perm_index_list).tolist()
    self.all_permutation_perm_index_list = all_permutation_perm_index_list
    self.all_permutation_option_list = all_permutation_option_list
    
    # amount of permutation 
    amount_of_perm = len(all_permutation_option_list)
    
    # freq to generate balance permutation distrebutions 
    frequency =  amount_of_sampels//amount_of_perm+1
    weighted_list = [element for element in self.all_permutation_option for _ in range(frequency)]
    
    # generate 2 list of permutation for ssl task when reauire to generate image with 2 different augmentation
    # the permutation augmentation is chosen uniformaly 
    self.perm_order_list = random.sample(weighted_list, amount_of_sampels)
    self.perm_order_list2 = random.sample(weighted_list, amount_of_sampels)
    
    # check if require to read image of sellect from data object      
    if data is None:
        read_image = True
    else:
        read_image = False
    self.read_image= read_image
    
    # set train and test augmentation
    if train :
        self.transform = transform_train
    else:
        self.transform = transform_test    
    
  def __len__(self):
    # get dataset size
    return self.index_list.size
  
    
 
  def getPositionEncoding(self, perm_order, d, n=10000, perm_index = -1):
       """
    
          Parameters
          ----------
          perm_order : permutation order
          d : vecor dimetion
          n : nominator for postion embedding equation 
          perm_index : perm index 
    
          Returns
          -------
          P : postion embedding vec 
          perm_label : onehotencoding for permutation 

       """
       
       # convert permutation into list 
       perm_order = perm_order.tolist()
       
       # get pemutation index  
       if perm_index == -1:
           k2 = calculate_permutation_position(tuple(perm_order))
       else:
           k2 = perm_index
           
       # get index of permutation 
       k = self.all_permutation_option_list.index(perm_order)
       k2 = k

        # generate postion embedding vec that will be filled 
       perm_label =  np.zeros((1,self.all_permutation_option.__len__()))
       
       # generate onehotencoding vec 
       perm_label[0,k]  = 1
       
       
       # generate postion embedding vec 
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
      """
      

      Parameters
      ----------
      image : image before any aumgnetation 
      Returns
      -------
      new_image : image after pemrutation augmnetation 
      perm_order : postion embedding vector 
      perm_label : permutation index 

      """
      
      # aaply augmentation 
      transform_image =  self.transform(image)
      
      # get amount of patch 
      amount_of_patch = self.amount_of_patch
      
      # get image size 
      dim_size = transform_image.shape[0]
      
      # get grid size 
      amount_of_rows = int(amount_of_patch**0.5)
      
      # check if grid is a divided of image size  
      is_devided =  dim_size%amount_of_rows != 0
      
      # calculate patch size 
      patch_row_size, patch_col_size = transform_image.shape[1]//amount_of_rows, transform_image.shape[2]//amount_of_rows
     
      # choose from first permutation list 
      if self.image_idx == 1:
            perm_order, perm_index = self.perm_order_list[self.idx]
      # choose from secound permutation list 
      else:
            perm_order, perm_index = self.perm_order_list2[self.idx]
            # validate that pemutation of image1 is not equal of image2
            if np.array_equal(perm_order, self.perm_order_list[self.idx][0]):
                perm_order, perm_index  = random.choice(self.perm_order_list2)

      
      # generate empty image 
      new_image = torch.zeros_like(transform_image)
      
      # initiate pointer for fiiling empty image 
      row = 0
      col = 0
      
      # this loop wiil fill empy image each time slice patch from the tranformed image and insert base the pointers 
      for index, i_perm in enumerate(perm_order):
          
          # sellect patch to silce pointers 
          i_permutation_row = i_perm//amount_of_rows
          i_permutation_col = i_perm%amount_of_rows
          
          row = index//amount_of_rows
          col = index%amount_of_rows
          
          # calculate where to put the patch 
          from_row = row*patch_row_size
          to_row = (row+1)*patch_row_size
          from_col = col*patch_col_size
          to_col =  (col+1)*patch_col_size
          
          # slice patch 
          try:
              
              from_row_perm = i_permutation_row*patch_row_size
              to_row_perm = (i_permutation_row+1)*patch_row_size
              from_col_perm = i_permutation_col*patch_col_size
              to_col_perm =  (i_permutation_col+1)*patch_col_size
              patch_image = transform_image[:,from_row_perm:to_row_perm,from_col_perm:to_col_perm] 
                            
          except:
              a=5 
              
          # set border size 
          border_size = 0
          
          # patch size 
          row_size, col_size = patch_image.shape[1::]

          # handling borders 
          if index == 0:   
              padd_val = random.choice([0])
              new_image += padd_val
          if row ==0 :
             patch_image[:,0:border_size,:]  = padd_val
          else:
              patch_image[:,0:border_size,:]  = padd_val
              
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
              patch_image[:,row_size-border_size::,:]  = padd_val
          else:
              patch_image[:,row_size-border_size::,:]  = padd_val
          
          # put patch base pointer 
          new_image[0:dim_size, from_row:to_row, from_col:to_col] = patch_image

      
      # base perm index generated postion embedding vector and permutation onehotvector 
      perm_order, perm_label = self.getPositionEncoding(perm_order, self.pe_dim, n=self.max_sentence_lenth, perm_index = perm_index)

      return new_image, perm_order, perm_label
  
  def get_permutation_image(self, image):
      """
      

      Parameters
      ----------
      image : image before any augmentaion

      Returns
      -------
      new_image : image after augmentation (permutation and aug) 
      perm_order : position embedding vec base permutation index 
      perm_label : onehotencodoing base permutation index 
      """
      
      # apply permutation aug and regular augmentation 
      if self.taske_name == 'perm' :
          new_image, perm_order, perm_label = self.permutatation_aug(image)              
      
      # apply regular augmentation without pemtuation augmentation  
      else:
          transform_image =  self.transform(image)
          
          # generate postion embedding and permutation label with dummy content
          perm_order = torch.empty((1,self.pe_dim))
          new_image = transform_image
          perm_label = np.zeros((1,len(self.all_permutation_option_list)))
      
      # convert to tensor 
      perm_order = torch.Tensor(perm_order)
      perm_label = torch.Tensor(perm_label)

      return new_image, perm_order, perm_label
  def generate_original_image_plot(self, image, axarr):
      """
      Parameters
      ----------
      image : image in float 
      axarr : figure index 
      Returns
      -------
      image figure 

      """
      np_image_0_1 =  np.array(image)
      np_image_0_1 = np_image_0_1.astype(np.uint8)
      axarr[0].imshow(np_image_0_1)
      axarr[0].title.set_text('image before transformation')
  def generate_transformed_image_plot(self, np_transform_image_0_1, axarr, image_index = 1):
      """
      

      Parameters
      ----------
      np_transform_image_0_1 : normelized image
      axarr : plot index 
      image_index : image index

      Returns
      -------
      plot image in axarr index after converting image into range 0-255

      """
      np_transform_image_0_1 *= np.array(self.stds)
      np_transform_image_0_1 += np.array(self.means)
      np_transform_image_0_1*=255
      np_transform_image_0_1 = np_transform_image_0_1.astype(np.uint8)
      axarr[image_index].imshow(np_transform_image_0_1)
      axarr[image_index].title.set_text(f'image after transformation {image_index}')
      
    
  def __getitem__(self, idx):
    """
      Parameters
      ----------
      idx : sample index, base on this sample index collect info about 
            sample data, and apply transformation on sample data. 

      Returns
      -------
      sample : sample of data consist of 
      1) transform_image - tensor of image after transformation 
      2) label - for supervised task 
      3) perm_order - postion embedding base of permutation index 
      4) label_name - the name of label 
      5) perm_label - onthotencoding base permutation index  


    """
    # get image
    data_df_row = self.data_df.iloc[self.index_list[idx]]
    # update index 
    self.idx = idx
    # get file name 
    label_file_name = data_df_row['class_index']
    # read image 
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
    # sellect image from dataset 
    else:
        image = self.data[self.index_list[idx]]
        image = self.pill_transform(image)
    
    # get label name and index 
    if self.data_name in ['train', 'val', 'test']:
        label_target = data_df_row['class_index']
        label_name = data_df_row['class_name']
    else:
        label_target = -1
        label_name = 'unknon'
    
    # generate onehotencoding base label index 
    label =  np.zeros(( self.amount_of_class))
    label[label_target] = 1
    label =  torch.Tensor(label)
    label = label.to(torch.float)
    
    # for supervised task require to generate single image, while in ssl 2 images
    if self.learning_type == 'supervised':
        desire_amount_of_images = 1
    else:
        desire_amount_of_images = 2
    
    # initiate generated image index 
    self.image_idx = 1
    
    # generate first image 
    new_image, perm_order,  perm_label = self.get_permutation_image(image)
    
    # increase generated image counter  
    self.image_idx += 1
    
    # check if require to generate the secound image for ssl case 
    if desire_amount_of_images > 1:
        # generate another image and collect them tothther 
        new_image2, prem_order2, perm_label2= self.get_permutation_image(image)
        new_image = torch.concatenate([new_image, new_image2])
        perm_order = torch.concatenate([perm_order, prem_order2])
        perm_label = torch.concatenate([perm_label, perm_label2])

    # set that new imasge is the image afer transformation 
    transform_image = new_image
    
    # define  sample
    sample = (transform_image, label, torch.Tensor(perm_order), label_name, perm_label)
    
    # generate deubg images 
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

def split_into_train_testval(all_train_df, test_df, random_state, val_split, train_split, data, learning_type):
    """
    Parameters
    ----------
    all_train_df : trian dataframe 
    test_df : trian dataframe
    random_state : set random state
    val_split : define split of train data into train and validation 
    train_split : decrease amount of training data 

    Returns
    -------
    index list for each data set for train\test\validation

    """
    if not data is None:
        try:
            train_index, val_index = train_test_split(np.arange(all_train_df.shape[0]), stratify = all_train_df['class_index'] ,  test_size=val_split, random_state=random_state)
        except:
            train_index, val_index = train_test_split(np.arange(all_train_df.shape[0]) ,  test_size=val_split, random_state=random_state)
            
        test_index = np.arange(test_df.shape[0])
    
        if train_split!=1:
            try:
                train_index, dummy = train_test_split(train_index, stratify = all_train_df['class_index'][train_index] ,  test_size=1-train_split, random_state=random_state)
                # test_index, dummy = train_test_split(test_index, stratify = test_df['class_index'][test_index] ,  test_size=1-train_split, random_state=random_state)
    
            except:
                train_index, dummy = train_test_split(train_index ,  test_size=1-train_split, random_state=random_state)
                # test_index, dummy = train_test_split(test_index ,  test_size=1-train_split, random_state=random_state) 
    else:
   
        if learning_type == 'self_supervised':
            train_val_index =  np.arange(all_train_df.shape[0])
            test_index = np.array([])
            train_index, val_index = train_test_split(train_val_index,  test_size=val_split, random_state=random_state)

        else:
            try:
                train_val_index, test_index = train_test_split(np.arange(all_train_df.shape[0]), stratify = all_train_df['class_index'] ,  test_size=0.5, random_state=random_state)
                train_index, val_index = train_test_split(train_val_index,  test_size=val_split, random_state=random_state)

            except:
                train_val_index, test_index = train_test_split(np.arange(all_train_df.shape[0]) ,  test_size=0.5, random_state=random_state)
                train_index, val_index = train_test_split(train_val_index,  test_size=val_split, random_state=random_state)

        if train_split!=1:
            try:
                train_index, dummy = train_test_split(train_index, stratify = all_train_df['class_index'][train_index] ,  test_size=1-train_split, random_state=random_state)
                # if  learning_type != 'self_supervised':
                #     test_index, dummy = train_test_split(test_index, stratify = all_train_df['class_index'][test_index] ,  test_size=1-train_split, random_state=random_state)
            except:
                train_index, dummy = train_test_split(train_index ,  test_size=1-train_split, random_state=random_state)
                # if  learning_type != 'self_supervised':
                #    test_index, dummy = train_test_split(test_index, stratify = all_train_df['class_index'][test_index] ,  test_size=1-train_split, random_state=random_state)
                
    return train_index, val_index, test_index


def initialize_dataloaders(all_train_df,  test_df, training_configuration, amount_of_patch = 4 ,batch_size=8, val_split=0.1, debug_batch_size=8, random_state=1001,
                           means = [0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225], image_size = 224, tb_writer = None, taske_name = 'perm',
                           learning_type = 'supervised', num_workers = 2, train_data = None, test_data = None, rand_choise = True,
                           pin_memory=False, orig_pe = True, train_split = 1):
    
    # parse parameters 
    batch_size = training_configuration.batch_size
    amount_of_patch = training_configuration.amount_of_patch
    taske_name = training_configuration.perm
    learning_type = training_configuration.learning_type
    num_workers = training_configuration.num_workers
    max_allowed_permutation = training_configuration.max_allowed_permutation
    pe_dim = training_configuration.pe_dim

    
    # get permutations 
    all_permutation_option = training_configuration.all_permutation_option
    
    # valid task list
    tasks_list = ['perm', 'no_perm']
    
    # validate that task is valid 
    if not taske_name in  tasks_list:
        assert False, 'task not defined'
    
    
    # set resize transform
    resize_transforms = transforms.Resize((image_size,image_size), interpolation = transforms.InterpolationMode.BILINEAR  )
    
    # initiate transformation list  
    transformations = []
    
    # use augmetation 
    if rand_choise:
        # if use permutation
        if taske_name == 'perm':
            min_scale = 0.6
            transformations  = [
                                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomApply([transforms.RandomResizedCrop(size = (96, 96), scale=(min_scale, 1.0))], p=1),
                                transforms.RandomResizedCrop(size = (image_size, image_size), scale=(min_scale, 1.0)),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.RandomApply([transforms.GaussianBlur(kernel_size= 3, sigma = (0.1, 2))],p = 0.5)   
                                ]
        # if do not use permutation
        else:
            min_scale = 0.2
        
            transformations  = [
                                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomApply([transforms.RandomResizedCrop(size = (96, 96), scale=(min_scale, 1.0))], p=1),
                                transforms.RandomResizedCrop(size = (image_size, image_size), scale=(min_scale, 1.0)),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.RandomApply([transforms.GaussianBlur(kernel_size= 3, sigma = (0.1, 2))],p = 0.5)   
                                ]
            
        # choose on aumgetation randomly base transformation list 
        rand_choise = transforms.RandomChoice(transformations)
    # dont use aumgentation
    else:
        # non augmnetation 
        rand_choise = transforms.RandomChoice( [transforms.RandomHorizontalFlip(p=0)])
    
    # define train transformation for the case of permutation and non-pemutation case  
    if taske_name == 'perm':
        data_transforms =   transforms.Compose([resize_transforms,
                                                rand_choise,
                                                resize_transforms,
                                                transforms.ToTensor(),
                                                transforms.Normalize(means, stds)
                                                ])
    elif taske_name == 'no_perm':
        data_transforms =   transforms.Compose([
                                                rand_choise,
                                                resize_transforms,
                                                transforms.ToTensor(),
                                                transforms.Normalize(means, stds)
                                                ])
    
    # define test transformation 
    test_transforms =  transforms.Compose([ resize_transforms,
                                            transforms.ToTensor(),
                                            transforms.Normalize(means, stds)])

    # get amount of class
    class_df =  all_train_df[['class_name', 'class_index']]
    class_df =  class_df.drop_duplicates(subset=['class_name'])

    # split data into train test and validation sample index 
    train_index, val_index, test_index= split_into_train_testval(all_train_df,test_df, random_state, val_split, train_split, train_data, learning_type)

    # set 3 DataSets fpr train\test\validation     
    X_train = MyDataset(all_train_df, class_df, data_transforms, test_transforms ,index_list = train_index, amount_of_patch=amount_of_patch, 
                        train = True, data_name = 'train', taske_name = taske_name, learning_type = learning_type, data=train_data,
                        all_permutation_option=all_permutation_option, orig_pe = orig_pe, pe_dim=pe_dim)
    X_val = MyDataset(all_train_df,class_df, data_transforms, test_transforms ,index_list = val_index, amount_of_patch=amount_of_patch, 
                      train = False, data_name = 'val', taske_name = taske_name, learning_type = learning_type, data=train_data, 
                      all_permutation_option=all_permutation_option, orig_pe = orig_pe, pe_dim=pe_dim)
    X_test = MyDataset(test_df, class_df, data_transforms, test_transforms , index_list = test_index, amount_of_patch=amount_of_patch,
                       train = False, data_name = 'test', taske_name = taske_name, learning_type = learning_type, data=test_data, 
                       all_permutation_option=all_permutation_option, orig_pe = orig_pe, pe_dim=pe_dim)

    # set 3 dataloader to each dataset
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size,shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = torch.utils.data.DataLoader(X_val, batch_size=batch_size,shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    debug_loader = torch.utils.data.DataLoader(copy.deepcopy(X_train), batch_size=debug_batch_size, shuffle=True, pin_memory=False)
    
    # print size of data-sets
    print(f'Train length = {train_loader.dataset.__len__()}, \
          val length = {val_loader.dataset.__len__()},  \
          test length = {test_loader.dataset.__len__()}')
          
          
    return train_loader, val_loader, test_loader, debug_loader