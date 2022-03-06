
import os.path
import torch
import torchvision.transforms.functional as tf
import torch.nn.functional as F
import random
from torchvision.transforms.transforms import RandomCrop, RandomResizedCrop
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from util import util

class MefDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--is_train', type=bool, default=True, help='whether in the training phase')
        parser.set_defaults(max_dataset_size=float("inf"), new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.fake_image_paths = []
        self.image_paths = []
        self.isTrain = opt.isTrain
        self.image_size = opt.crop_size
        
        if opt.isTrain==True:
            print('loading training file')
            self.trainfile = opt.dataset_root+'Dataset_Part1_resize/'+'part1_train.txt'
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        name = line.rstrip().split('.')
                        self.image_paths.append(os.path.join(opt.dataset_root,'Dataset_Part1_resize/',name[0]))
            self.trainfile = opt.dataset_root+'Dataset_Part2_resize/'+'part2_train.txt'
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        name = line.rstrip().split('.')
                        self.image_paths.append(os.path.join(opt.dataset_root,'Dataset_Part2_resize/',name[0]))           
        elif opt.isTrain==False:
            print('loading test file')
            self.trainfile = opt.dataset_root+'Dataset_Part1_resize/'+'part1_test.txt'
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        name = line.rstrip().split('.')
                        self.image_paths.append(os.path.join(opt.dataset_root,'Dataset_Part1_resize/',name[0]))
            self.trainfile = opt.dataset_root+'Dataset_Part2_resize/'+'part2_test.txt'
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        name = line.rstrip().split('.')
                        self.image_paths.append(os.path.join(opt.dataset_root,'Dataset_Part2_resize/',name[0])) 
        transform_list = [
            # transforms.RandomCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (1, 1, 1))
        ]
        self.transforms = transforms.Compose(transform_list)
    def __getitem__(self, index):
        path = self.image_paths[index]
        files = os.listdir(path)

        files.sort(key= lambda x:int(x[:-4]))
        if self.isTrain:
            max_file = files[-1]
            min_file = files[0]
        else:
            max_file = files[-1]
            min_file = files[0]
        
        u_path = os.path.join(path,min_file)
        o_path = os.path.join(path,max_file)
        file_name_path = path+".JPG"
        name_parts=file_name_path.split('/')
        target_path = file_name_path.replace(name_parts[-1],'Label/'+name_parts[-1])
        if not os.path.exists(target_path):
            target_path = target_path.replace(".JPG",".PNG")
        fake_u = Image.open(u_path).convert('RGB')
        fake_o = Image.open(o_path).convert('RGB')
        real = Image.open(target_path).convert('RGB')
        if np.random.rand() > 0.5 and self.isTrain:
            fake_u, fake_o, real = tf.hflip(fake_u), tf.hflip(fake_o), tf.hflip(real)
        fake_u = self.transforms(fake_u)
        fake_o = self.transforms(fake_o)
        real = self.transforms(real)

        return {'fake_u': fake_u, 'fake_o': fake_o, 'real': real,'img_path':path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)