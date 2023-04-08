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


def parse_train_data(images_path_list):
    class_name_list = list(map(lambda x: os.path.basename(x).split('.')[0], images_path_list))
    data_class_df = pd.DataFrame([['cat', 0],['dog',1]], columns = ['class_name', 'class_index'])
    train_df = pd.DataFrame(class_name_list, columns = ['class_name'])
    train_df['image_path'] = images_path_list
    train_df = pd.merge(train_df, data_class_df, how = 'left', on = ['class_name'])
    return train_df
def parse_test_data(images_path_list):
    test_df = pd.DataFrame()
    test_df['image_path'] = images_path_list
    return test_df



class MyDataset(Dataset):

  def __init__(self, data_df, class_df,  transform_train, transform_test , amount_of_patch = 4, train = True, data_name = 'train',
               debug = False, max_debug_image_allowed = 0, means = [0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225], taske_name = 'perm'):
    
      
    self.taske_name = taske_name
    self.amount_of_patch = amount_of_patch
    self.means = means
    self.stds = stds
    self.debug = debug
    self.debug_image_idx = 0
    self.max_debug_image_allowed = max_debug_image_allowed
    self.train = train
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
    return self.data_df.shape[0]
  
  def permutatation_aug(self, transform_image):
      amount_of_patch = self.amount_of_patch
      dim_size = transform_image.shape[0]
      amount_of_rows = int(amount_of_patch**0.5)
      patch_row_size, patch_col_size = transform_image.shape[1]//amount_of_rows, transform_image.shape[2]//amount_of_rows
      patch_array = patchify(transform_image.numpy(), (dim_size, patch_row_size, patch_col_size), patch_row_size)
      prem_order  = random.sample(range(amount_of_patch), amount_of_patch)
      new_image = torch.zeros_like(transform_image)
      row = 0
      col = 0
      for index, i_perm in enumerate(prem_order):
          i_perm_row = i_perm//amount_of_rows
          i_perm_col = i_perm%amount_of_rows
          
          row = index//amount_of_rows
          col = index%amount_of_rows
          
          from_row = row*patch_row_size
          to_row = (row+1)*patch_row_size
          from_col = col*patch_col_size
          to_col =  (col+1)*patch_col_size
          new_image[0:dim_size, from_row:to_row, from_col:to_col] = torch.Tensor(patch_array[0][i_perm_row, i_perm_col])
      return new_image, prem_order
  
    
  def __getitem__(self, idx):
    # get image
    
    data_df_row = self.data_df.iloc[idx]
    label_file_name = data_df_row['class_index']
    image_path = data_df_row['image_path']

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
    image = Image.open(image_path) 
    
    transform_image =  self.transform(image)
    
    
    if self.taske_name == 'perm':
        new_image, prem_order = self.permutatation_aug(transform_image)
    else:
        prem_order = torch.empty(self.amount_of_patch)
        new_image = transform_image
    transform_image = new_image
    # set sample
    sample = (transform_image, label, torch.Tensor(prem_order), label_name)
    
    if self.debug:
      
        np_transform_image_0_1 =  transform_image.numpy()
        np_transform_image_0_1 = np_transform_image_0_1.transpose(1,2,0)

        np_transform_image_0_1 *= np.array(self.stds)
        np_transform_image_0_1 += np.array(self.means)
        np_transform_image_0_1*=255
        np_image_0_1 =  np.array(image)
        np_image_0_1 = np_image_0_1.astype(np.uint8)
        np_transform_image_0_1 = np_transform_image_0_1.astype(np.uint8)
            
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(np_image_0_1)
        axarr[0].title.set_text('image before transformation')
        axarr[1].imshow(np_transform_image_0_1)
        axarr[1].title.set_text(f'image after transformation')     
        f.suptitle(f'Class is {label_name}') # or plt.suptitle('Main title')


    return sample


def initialize_dataloaders(all_train_df,  test_df, amount_of_patch = 4 ,batch_size=8, val_split=0.1, debug_batch_size=8, random_state=1001,
                           means = [0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225], image_size = 224, tb_writer = None, taske_name = 'perm' ):
    
    
    tasks_list = ['perm', 'no_perm']
    if not taske_name in  tasks_list:
        assert False, 'task not defined'
    if taske_name == 'perm':
        data_transforms =   transforms.Compose([transforms.Resize((image_size,image_size)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(means, stds)])
    elif taske_name == 'no_perm':
        data_transforms =   transforms.Compose([transforms.Resize((image_size,image_size)),
                                                transforms.RandomChoice( [
                                                                          transforms.RandomCrop(size=(image_size, image_size)),
                                                                          transforms.RandomHorizontalFlip(p=0.5),
                                                                          transforms.RandomAffine(degrees=(-5, 5), 
                                                                                                  translate=(0, 0.2), 
                                                                                                  scale=(0.6, 1)),
                                                                          transforms.RandomVerticalFlip(p=0.5)]),
                                                transforms.ToTensor(),
                                                transforms.Normalize(means, stds)])
    
    test_transforms =  transforms.Compose([transforms.Resize((image_size,image_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(means, stds)])

    # get amount of class
    class_df =  all_train_df[['class_name', 'class_index']]
    class_df =  class_df.drop_duplicates(subset=['class_name'])
    
    
    train_df, val_df = train_test_split(all_train_df, stratify = all_train_df['class_name'],  test_size=val_split, random_state=random_state)


    X_train = MyDataset(train_df, class_df, data_transforms, test_transforms ,amount_of_patch=amount_of_patch, train = True, data_name = 'train', taske_name = taske_name)
    X_val = MyDataset(val_df,class_df, data_transforms, test_transforms ,amount_of_patch=amount_of_patch,  train = False, data_name = 'val', taske_name = taske_name)
    X_test = MyDataset(test_df, class_df, data_transforms, test_transforms , amount_of_patch=amount_of_patch, train = False, data_name = 'test', taske_name = taske_name)

       
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(X_val, batch_size=batch_size,shuffle=False)
    test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=False)
    debug_loader = torch.utils.data.DataLoader(copy.deepcopy(X_train), batch_size=debug_batch_size, shuffle=True)
    # if not tb_writer is None and 0:
    #     add_data_embedings(val_loader, tb_writer, n=300)

    return train_loader, val_loader, test_loader, debug_loader