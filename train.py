import os
import torch
import torch.distributed as dist
from util.misc import launch_job
from train_net import train

from options.train_options import TrainOptions
    
def main():
    cfg = TrainOptions().parse()   # get training options
    cfg.NUM_GPUS = torch.cuda.device_count()
    cfg.batch_size = int(cfg.batch_size / max(1, cfg.NUM_GPUS))
    cfg.phase = 'train'
    launch_job(cfg=cfg, init_method=cfg.init_method, func=train)


if __name__=="__main__":
    main()