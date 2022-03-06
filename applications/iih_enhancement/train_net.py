import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from util import distributed as du

import time
from collections import OrderedDict
from data import create_dataset
from data import shuffle_dataset
from models import create_model
from util.visualizer import Visualizer
from util.evaluation import evaluation
from util import html,util
from util.visualizer import save_images

def train(cfg):
    #init
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    #init dataset
    dataset = create_dataset(cfg)  # create a dataset given cfg.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(cfg)      # create a model given cfg.model and other options
    model.setup(cfg)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(cfg)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    # cur_device = torch.cuda.current_device()
    is_master = du.is_master_proc(cfg.NUM_GPUS)
    for epoch in range(cfg.epoch_count, cfg.niter + cfg.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        if is_master:
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        shuffle_dataset(dataset, epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            if is_master:
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % cfg.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                    iter_data_time = time.time()
            visualizer.reset()
            total_iters += cfg.batch_size
            epoch_iter += cfg.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % cfg.display_freq == 0 and is_master:   # display images on visdom and save images to a HTML file
                save_result = total_iters % cfg.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            losses = model.get_current_losses()
            if cfg.NUM_GPUS > 1:
                losses = du.all_reduce(losses)
            if total_iters % cfg.print_freq == 0 and is_master:    # print training losses and save logging information to the disk
                t_comp = (time.time() - iter_start_time) / cfg.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if cfg.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
            if total_iters % cfg.save_latest_freq == 0 and is_master:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if cfg.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                
        if epoch % cfg.save_epoch_freq == 0 and is_master:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            if cfg.save_iter_model and epoch>=30:
                model.save_networks(epoch)
        if is_master:
            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, cfg.niter + cfg.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.


def test(cfg):
    dataset = create_dataset(cfg)  # create a dataset given cfg.dataset_mode and other options
    model = create_model(cfg)      # create a model given cfg.model and other options
    model.setup(cfg)               # regular setup: load and print networks; create schedulers

    # create a website
    web_dir = os.path.join(cfg.results_dir, cfg.name, '%s_%s' % (cfg.phase, cfg.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (cfg.name, cfg.phase, cfg.epoch))
    if cfg.eval:
        model.eval()
    ismaster = du.is_master_proc(cfg.NUM_GPUS)

    num_image = 0
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        
        img_path = model.get_image_paths()     # get image paths # Added by Mia
        if i % 5 == 0 and ismaster:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        visuals_ones = OrderedDict()
        for j in range(len(img_path)):
            img_path_one = []
            for label, im_data in visuals.items():
                visuals_ones[label] = im_data[j:j+1, :, :, :]
            img_path_one.append(img_path[j])
            save_images(webpage, visuals_ones, img_path_one, aspect_ratio=cfg.aspect_ratio, width=cfg.display_winsize)
            num_image += 1
            visuals_ones.clear()

    webpage.save()  # save the HTML


