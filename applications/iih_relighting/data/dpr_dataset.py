"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os.path
import torch
import torchvision.transforms.functional as tf
import torch.nn.functional as F
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
from scipy import io
import torchvision.transforms as transforms
from util import util

class DPRDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--is_train', type=bool, default=True, help='whether in the training phase')
        parser.set_defaults(max_dataset_size=float("inf"), new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.image_paths = []
        self.isTrain = opt.isTrain
        self.image_size = opt.crop_size
        self.real_image_paths = []
        if opt.isTrain==True:
            print('loading training file')
            self.trainfile = opt.dataset_root+'train.lst'
            with open(self.trainfile,'r') as f:
                for line in f.readlines():
                    file_names = line.rstrip()
                    file_name_arr = file_names.split(" ")
                    self.image_paths.append(os.path.join(opt.dataset_root,'DPR_dataset',file_name_arr[0],file_name_arr[1]))
                    self.real_image_paths.append(os.path.join(opt.dataset_root,'DPR_dataset',file_name_arr[0],file_name_arr[2]))
        elif opt.isTrain==False:
            #self.real_ext='.jpg'
            print('loading test file')
            # self.trainfile = opt.dataset_root+'test.lst'
            self.trainfile = opt.dataset_root+'randomdpr_test.txt'
            with open(self.trainfile,'r') as f:
                for line in f.readlines():
                    file_names = line.rstrip()
                    file_name_arr = file_names.split(" ")
                    self.image_paths.append(os.path.join(opt.dataset_root,'DPR_dataset',file_name_arr[0],file_name_arr[1]))
                    self.real_image_paths.append(os.path.join(opt.dataset_root,'DPR_dataset',file_name_arr[0],file_name_arr[2]))
            
        # get the image paths of your dataset;
          # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        transform_list = [
            transforms.Resize([256,256]),
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (1, 1, 1))
        ]
        self.transforms = transforms.Compose(transform_list)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        path = self.image_paths[index]
        real_path = self.real_image_paths[index]
        fake_path = path
        fake_light_path = fake_path.replace('_0', "_light_0").replace(".png",".txt")
        real_light_path = real_path.replace('_0', "_light_0").replace(".png",".txt")
        fake = Image.open(fake_path).convert('RGB')
        real = Image.open(real_path).convert('RGB')

        fake_light = np.loadtxt(fake_light_path)
        fake_light = fake_light[0:9]
        fake_light = np.squeeze(fake_light).astype(np.float32)
        fake_light = torch.from_numpy(fake_light)

        real_light = np.loadtxt(real_light_path)
        real_light = real_light[0:9]
        real_light = np.squeeze(real_light).astype(np.float32)
        real_light = torch.from_numpy(real_light)
        fake = self.transforms(fake).to(torch.float32)
        real = self.transforms(real).to(torch.float32)

        real_path_tmp = real_path[-7:]
        path = path[:-4]+real_path_tmp

        return {'fake_light': fake_light, 'real_light':real_light, 'fake': fake, 'target':real, 'real':real, 'img_path':path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)


