import os 
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def set_matplotlib():
    # set plot fonts
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.close('all')
    sns.set_style("white")
def get_all_images_from_specific_folder(folder_path, ending = 'jpg'):
    glob_string = os.path.join(folder_path, '*.' + ending)
    images_path_list = glob.glob(glob_string)
    return images_path_list


def generate_hitogram_base_dataframe_column(df, column_name):
    plt.figure()
    sns.histplot(data=df, x=column_name)
    plt.grid()
    plt.title('histogram base column: ' + column_name)


