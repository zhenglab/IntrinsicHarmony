import torch
import itertools
import torch.nn.functional as F
from util import distributed as du
from .base_model import BaseModel
from util import util
from . import relighting_networks as networks
from . import networks as network_init
import util.ssim as ssim


class IIHBaseLTModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(norm='instance', netG='base_lt', dataset_mode='dpr')
        parser.add_argument('--action', type=str, default='relighting', help='weight for L1 loss')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_R_gradient', type=float, default=10., help='weight for R gradient loss')
            parser.add_argument('--lambda_ssim', type=float, default=50., help='weight for L L2 loss')
            parser.add_argument('--lambda_I_smooth', type=float, default=1., help='weight for L L2 loss')
            parser.add_argument('--lambda_I_L2', type=float, default=10., help='weight for L L2 loss')
            parser.add_argument('--lambda_ifm', type=float, default=100, help='weight for pm loss')
            parser.add_argument('--lambda_L', type=float, default=100.0, help='weight for L1 loss')
            
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.opt = opt
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G','G_L1','G_R','G_I_L2','G_I_smooth',"G_L"]
            
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['harmonized','real','fake','reflectance','illumination']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']
        self.opt.device = self.device
        self.netG = networks.define_G(opt.netG, opt.init_type, opt.init_gain, self.opt)
        self.cur_device = torch.cuda.current_device()
        self.ismaster = du.is_master_proc(opt.NUM_GPUS)
        if self.ismaster:
            print(self.netG)  
        
        if self.isTrain:
            util.saveprint(self.opt, 'netG', str(self.netG))  
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionSSIM = ssim.SSIM()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):

        self.fake = input['fake'].to(self.device)
        self.real = input['real'].to(self.device)
        if input['target'] is not None:
            self.target = input['target'].to(self.device)
        if input['fake_light'] is not None:
            self.light_fake = input['fake_light'].to(self.device)
        if input['real_light'] is not None:
            self.light_real = input['real_light'].to(self.device)
        self.image_paths = input['img_path']
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            self.harmonized, self.reflectance, self.illumination, self.light_gen_fake = self.netG(self.fake, self.light_real)
        else:
            if self.opt.action == "relighting":
                self.harmonized, self.reflectance, self.illumination, self.light_gen_fake = self.netG(self.fake, isTest=True, light=self.light_real)
            else:
                self.harmonized, self.reflectance, self.illumination, self.light_gen_fake = self.netG(self.fake, isTest=True, target=self.target)
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G_L1 = self.criterionL1(self.harmonized, self.real)*self.opt.lambda_L1
        self.loss_G_R = (self.gradient_loss(self.reflectance, self.fake)+self.gradient_loss(self.reflectance, self.real))*self.opt.lambda_R_gradient
        self.loss_G_I_smooth = util.compute_smooth_loss(self.illumination)*self.opt.lambda_I_smooth
        self.loss_G_I_L2 = self.criterionL2(self.illumination, self.real)*self.opt.lambda_I_L2
        self.loss_G_L = self.criterionL2(self.light_gen_fake, self.light_fake)*self.opt.lambda_L
        # assert 0
        self.loss_G = self.loss_G_L1 + self.loss_G_R + self.loss_G_I_smooth + self.loss_G_I_L2 + self.loss_G_L
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def gradient_loss(self, input_1, input_2):
        g_x = self.criterionL1(util.gradient(input_1, 'x'), util.gradient(input_2, 'x'))
        g_y = self.criterionL1(util.gradient(input_1, 'y'), util.gradient(input_2, 'y'))
        return g_x+g_y
        
    def __compute_kl(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss
   
