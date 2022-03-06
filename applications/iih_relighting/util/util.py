"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
# from PIL import Image
from PIL import Image,ImageDraw, ImageFont
import matplotlib.font_manager as fm # to create font
import torch.nn as nn
import os
import torch.nn.functional as F
from util.tools import *

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 0) / 1.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    image_numpy = np.clip(image_numpy, 0,255)
    return image_numpy.astype(imtype)

def retry_load_images(image_paths, retry=10, backend="pytorch"):
    """
    This function is to load images with support of retrying for failed load.

    Args:
        image_paths (list): paths of images needed to be loaded.
        retry (int, optional): maximum time of loading retrying. Defaults to 10.
        backend (str): `pytorch` or `cv2`.

    Returns:
        imgs (list): list of loaded images.
    """
    for i in range(retry):
        imgs = []
        for image_path in image_paths:
            with PathManager.open(image_path, "rb") as f:
                img_str = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_str, flags=cv2.IMREAD_COLOR)
            imgs.append(img)

        if all(img is not None for img in imgs):
            if backend == "pytorch":
                imgs = torch.as_tensor(np.stack(imgs))
            return imgs
        else:
            logger.warn("Reading failed. Will retry.")
            time.sleep(1.0)
        if i == retry - 1:
            raise Exception("Failed to load images {}".format(image_paths))

def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    # image_pil.save(image_path,quality=100) #added by Mia (quality)
    image_pil.save(image_path,quality=95) #added by Mia (quality)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def calc_unmask_mean(feat,mask, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C, H, W = size
    feat_unmask = feat*(1-mask)
    feat_unmask_sum = feat.view(N, C, -1).sum(dim=2)
    mask_pixel_sum = mask.view(mask.size(0), mask.size(1), -1).sum(dim=2)
    feat_unmask_mean = feat_unmask_sum.div(H*W-mask_pixel_sum).view(N, C, 1, 1)
    return feat_unmask_mean

def saveprint(opt, name, message):
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, '{}_opt.txt'.format(name))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

def clip_by_tensor(t,t_min,t_max=None):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t=t.float()
    t_min=t_min.float()
    
 
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    if t_max is not None:
        t_max=t_max.float()
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss

def gradient(input_tensor, direction):
    # input_tensor = input_tensor.permute(0, 3, 1, 2)
    b,c,h, w = input_tensor.size()

    smooth_kernel_x = torch.reshape(torch.Tensor([[0., 0.], [-1., 1.]]), (1, 1, 2, 2)).repeat(1,c,1,1).to(input_tensor.get_device())
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)

    assert direction in ['x', 'y']
    if direction == "x":
        kernel = smooth_kernel_x
    else:
        kernel = smooth_kernel_y

    out = F.conv2d(input_tensor, kernel, padding=(1, 1))
    out = torch.abs(out[:, :, 0:h, 0:w])
    return out
    # return out.permute(0, 2, 3, 1)

def gradient_sobel(input_tensor, direction):
    # input_tensor = input_tensor.permute(0, 3, 1, 2)
    h, w = input_tensor.size()[2], input_tensor.size()[3]

    smooth_kernel_x = torch.reshape(torch.Tensor([[-1., 0., 1], [-2.,0, 2.], [-1, 0, 1]]), (1, 1, 3, 3)).to(input_tensor.get_device())
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)

    assert direction in ['x', 'y']
    if direction == "x":
        kernel = smooth_kernel_x
    else:
        kernel = smooth_kernel_y

    out = F.conv2d(input_tensor, kernel, padding=(1, 1))
    # out = torch.abs(out[:, :, 0:h, 0:w])
    return out


def ave_gradient(input_tensor, direction):
    return (F.avg_pool2d(gradient(input_tensor, direction), 3, stride=1, padding=1))
    # return (F.avg_pool2d(gradient(input_tensor, direction).permute(0, 3, 1, 2), 3, stride=1, padding=1))\
    #     .permute(0, 2, 3, 1)



def smooth(input_l, input_r):
    # input_r = torch.cat([input_r,input_r,input_r],dim=1)
    # rgb_weights = torch.Tensor([0.2989, 0.5870, 0.1140])
    # input_r = torch.tensordot(input_r, rgb_weights, dims=([-1], [-1]))
    # input_r = torch.unsqueeze(input_r, -1)
    return torch.norm(gradient_sobel(input_l, 'x'), 1) + torch.norm(gradient_sobel(input_l, 'y'), 1)
    return torch.mean(
        gradient(input_l, 'x') * torch.exp(-10 * ave_gradient(input_r, 'x')) +
        gradient(input_l, 'y') * torch.exp(-10 * ave_gradient(input_r, 'y'))
    )
def calImageGradient(image):
    if image.size(1) >1:
        image = rgbtogray(image)
    gradient_x = gradient(image, 'x')
    gradient_y = gradient(image, 'y')
    gradient_i = gradient_x + gradient_y
    return gradient_i

def calRobustRetinexG(image):
    
    gradient_i = calImageGradient(image)
    k = 1 + 10 * torch.exp(-torch.abs(gradient_i).div(10))
    return gradient_i * k

def rgbtogray(image):
    # image1 = torch.rand(1,3,2,2)
    rgb_weights = torch.Tensor([0.2989, 0.5870, 0.1140]).to(image.get_device())
    input_r = torch.tensordot(image, rgb_weights, dims=([-3], [-1]))
    input_r = input_r.unsqueeze(-3)
    return input_r
    

def compute_smooth_loss(pred_disp):
        def gradient(pred):
            D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy
        dx, dy = gradient(pred_disp)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        return torch.mean(torch.abs(dx2)) + \
               torch.mean(torch.abs(dxdy)) + \
               torch.mean(torch.abs(dydx)) + \
               torch.mean(torch.abs(dy2))

def exposure_loss(gen, mask):
    mask_image_mean = calc_unmask_mean(gen, mask)
    mean = F.adaptive_avg_pool2d(gen, 16)
    d = torch.mean(torch.pow(mean- mask_image_mean,2))
    return d

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    # images = torch.nn.ZeroPad2d(paddings)(images)
    images = torch.nn.ReflectionPad2d(paddings)(images)
    return images

def extract_image_patches(images, ksizes, strides, rates, padding='same',paddingsize=0):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pad = nn.ReflectionPad2d(paddingsize)
        images = pad(images)
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks

def sirfs_smooth_loss(images):
    n,c,h,w = images.size()
    k = 5
    patches = extract_image_patches(images, ksizes=[k, k],
                                      strides=[1,1],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L]
    patches = patches.view(n,c,-1, k*k)
    patches_sum = torch.sum(patches, dim=3, keepdim=True)
    patches_sum = patches_sum.view(n, -1, h, w)
    reduces = torch.abs(images*25-patches_sum)
    loss = reduces.mean()
    # assert 1==1
    return loss
    # assert 1

def save_layer_feature(style_feat, in_featuremap, out_featuremap, save_path, file_name, mean, std):
    paddding = nn.ZeroPad2d(5)
    in_featuremap = paddding(in_featuremap).tanh()/2+0.5
    out_featuremap = paddding(out_featuremap).tanh()/2+0.5
    style_feat = paddding(style_feat).tanh()/2+0.5

    
    b, c, h, w = in_featuremap.size()
    
    # features = feature_numpy[0]
    result = Image.new('RGB', (w*c, h*3+60))
    
    mean = mean.view(3, b, c)
    std = std.view(3, b, c)
    content_mean = mean[0]
    content_std = std[0]
    style_mean = mean[1]
    style_std = std[1]
    adain_mean = mean[2]
    adain_std = std[2]

    fontsize = 10
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),fontsize)

    for i in range(c):
        in_feature = style_feat[0:1, i:i+1, :, :]
        in_feature_numpy = tensor2im(in_feature)
        in_image_pil = Image.fromarray(in_feature_numpy)
        result.paste(in_image_pil, box=(w*i, 0))

        in_feature = in_featuremap[0:1, i:i+1, :, :]
        in_feature_numpy = tensor2im(in_feature)
        in_image_pil = Image.fromarray(in_feature_numpy)
        result.paste(in_image_pil, box=(w*i, h))

        out_feature = out_featuremap[0:1, i:i+1, :, :]
        out_max = out_feature[0][0].max()
        out_feature_numpy = tensor2im(out_feature)
        np_max = np.max(out_feature_numpy)
        location = np.where(out_feature_numpy==np_max)
        out_image_pil = Image.fromarray(out_feature_numpy)
        result.paste(out_image_pil, box=(w*i, h*2))
        
        draw = ImageDraw.Draw(result)
        color = "#FF0000"
        string = str(round(content_mean[0, i].cpu().item(),2))+', '+str(round(content_std[0, i].cpu().item(),2))
        draw.text((w*i, h*3+4), string, font=font, fill=color, spacing=0, align='left')
        color = "#00FF00"
        string = str(round(style_mean[0, i].cpu().item(),2))+', '+str(round(style_std[0, i].cpu().item(),2))
        draw.text((w*i, h*3+4+20), string, font=font, fill=color, spacing=0, align='left')
        color = "#FFFFFF"
        string = str(round(adain_mean[0, i].cpu().item(),2))+', '+str(round(adain_std[0, i].cpu().item(),2))
        draw.text((w*i, h*3+4+40), string, font=font, fill=color, spacing=0, align='left')

    
    save_name = os.path.join(save_path, str(i)+'.jpg')
    result.save(save_name,  quality=100)

def save_feature_value(feature, save_path):
    h, w = feature.size()
    values = []
    for i in range(h):
        value = ''
        for j in range(w):
            value += str(feature[i][j].cpu().item()) +', '
        values.append(value)
    file=open(save_path,'w') 
    for line in values:
        file.write(str(line)+"\n")
    file.close() 

def gredient_xy(images):
    gradient_x = gradient(images, 'x')
    gradient_y = gradient(images, 'y')
    gredient_images = gradient_x + gradient_y
    return gredient_images


def images_patches_L2_scores(images, fg_mask, lamda=50, ksize=3, stride=1,size=64):
    fg_mask = F.interpolate(fg_mask, size=[size, size], mode="nearest")
    images = F.interpolate(images, size=[size, size], mode="nearest")
    # b, dims, h, w = images.size()
    bg = images * (1-fg_mask)
    fg = images * fg_mask
    attScore, DS = patches_distance_L2(fg, bg, ksize=ksize, stride=stride)
    return attScore, DS

def patches_distance_L2(images, fg_mask, ksize=3, stride=1, lamda=10):
    b, dims, h, w = images.size()
    fg_mask = F.interpolate(fg_mask, size=[h, w], mode="nearest")
    bg = images * (1-fg_mask)
    fg = images * fg_mask
    # bg = images
    # fg = images
    patch1 = extract_image_patches(bg, ksizes=[ksize, ksize],
                                strides=[stride, stride],
                                rates=[1, 1],
                                padding='same',
                                paddingsize=1)     #1 2304 1024  [N, C*k*k, L]

    
    patch1 = patch1.view(b, dims, ksize, ksize, -1)
    patch1 = patch1.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]

    patch2 = extract_image_patches(fg, ksizes=[ksize, ksize],
                                strides=[stride, stride],
                                rates=[1, 1],
                                padding='same',
                                paddingsize=1) #1 2304 1024  [N, C*k*k, L]
    patch2 = patch2.view(b, -1, h, w) #[N C*k*k H W]
    ACL = []

    # patch3 = extract_image_patches(fg_mask, ksizes=[ksize, ksize],
    #                             strides=[stride, stride],
    #                             rates=[1, 1],
    #                             padding='same',
    #                             paddingsize=1) #1 2304 1024  [N, C*k*k, L]
    # mask_patches = patch3.view(b, ksize*ksize, -1).permute(0,2,1)
    # mask_patches = torch.sum(mask_patches, dim=2)
    
    for ib in range(b):
        # mask_patch = mask_patches[ib,:]
        # mask_fg_patches = torch.nonzero(mask_patch).squeeze(1)

        k1 = patch1[ib, :, :, :, :]   #[L, C, k, k]
        k1d = reduce_sum(torch.pow(k1, 2), axis=[1,2,3], keepdim=True).view(k1.size(0), 1, 1)  #L 1 1 
        ww = patch2[ib, :, :]   #[C*k*k H W]
        wwd = torch.sum(torch.pow(ww, 2), dim=0, keepdim=True) #[1 h w]
        ft = ww.unsqueeze(0) #[1 C h w]

        k1 = k1.view(k1.size(0), -1, 1, 1)
        CS = F.conv2d(ft, k1, stride=1)   # [1, L, H, W]

        tt = k1d + wwd

        DS1 = tt.unsqueeze(0) - 2*CS  # [1, L, H, W]

        DS2 = (DS1 - torch.mean(DS1, dim=1, keepdim=True)) / reduce_std(DS1, [1], True)
        DS2 = -1 * torch.tanh(DS2)

        CA = F.softmax(lamda * DS2, dim=1)
        # CA_clean = torch.zeros_like(CA).to(CA.get_device())
        # for i in enumerate(mask_fg_patches):
        #     m = i[1]
        #     CA_clean[0, m, :, :] = CA[0, m, :, :]
        # CA_clean = zeros.scatter(1, mask_fg_patches, CA)
        # CA_max,_ = torch.max(CA.view(1,CA.size(1),-1), dim=2)
        # CA_max = CA_max.view(1,CA.size(1),1,1)

        # CA = torch.where(CA[:,:,:,:]<CA_max[:,:], torch.zeros_like(CA).to(CA.get_device()), CA[:,:,:,:])
        # CA_1 = torch.where(CA[:,:,:,:]<CA_max[:,:], torch.zeros_like(CA).to(CA.get_device()), torch.ones_like(CA).to(CA.get_device()))
        # CA_max,_ = torch.max(CA.view(1,CA.size(1),-1), dim=1)
        # torch.full_like(x, 5)
        if ib == 0:
            CA_batchs = CA
            # CA_batchs_1 = CA_1
        else:
            CA_batchs = torch.cat([CA_batchs, CA], dim=0)
            # CA_batchs_1 = torch.cat([CA_batchs_1, CA_1], dim=0)
    return CA_batchs


def patches_distance_SSIM(images, fg_mask, ksize=4, stride=2, lamda=10):
    b, dims, h, w = images.size()
    bg = images * (1-fg_mask)
    fg = images * fg_mask
    h=h//stride
    w=w//stride
    patch1 = extract_image_patches(bg, ksizes=[ksize, ksize],
                                strides=[stride, stride],
                                rates=[1, 1],
                                padding='valid',
                                paddingsize=1)     #1 2304 1024  [N, C*k*k, L]

    patch1 = patch1.view(b, dims, ksize, ksize, -1)
    patch1 = patch1.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]

    patch2 = extract_image_patches(fg, ksizes=[ksize, ksize],
                                strides=[stride, stride],
                                rates=[1, 1],
                                padding='valid',
                                paddingsize=1) #1 2304 1024  [N, C*k*k, L]
    patch2 = patch2.view(b, -1, h, w) #[N C*k*k H W]
    ACL = []

    C2 = 0.03**2
    C1 = 0.01**2
    patch_count = dims*ksize*ksize
    for ib in range(b):

        k1 = patch1[ib, :, :, :, :]   #[L, C, k, k]
        k1d = reduce_sum(torch.pow(k1, 2), axis=[1,2,3], keepdim=True).view(k1.size(0), 1, 1)  #L 1 1 
        k1_mu = torch.mean(k1.view(k1.size(0),-1), dim=1, keepdim=True)
        k1_mu_sq = torch.pow(k1_mu.unsqueeze(-1),2)

        ww = patch2[ib, :, :]   #[C*k*k H W]
        wwd = torch.sum(torch.pow(ww, 2), dim=0, keepdim=True) #[1 h w]
        ww_mu = torch.mean(ww, dim=0, keepdim=True)
        ww_mu_sq = torch.pow(ww_mu,2)

        ft = ww.unsqueeze(0) #[1 C h w]
        k1 = k1.view(k1.size(0), -1, 1, 1)
        wk_mat = F.conv2d(ft, k1, stride=1)   # [1, L, H, W]

        ww_mu_k = ww_mu.unsqueeze(0) #[1 1 h w]
        k1_mu_k = k1_mu.view(k1_mu.size(0), 1, 1, 1)
        k1_ww_mu = F.conv2d(ww_mu_k, k1_mu_k)

        sigma12 = wk_mat/patch_count-k1_ww_mu

        sigma_k = k1d/patch_count-k1_mu_sq
        sigma_w = wwd/patch_count-ww_mu_sq

        ssim_s = (2*sigma12+C2)/(sigma_k+sigma_w+C2)

        # ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        # ssim_c = (2*k1_ww_mu+C1)/(k1_mu_sq+ww_mu_sq+C2)
        # ssim_s = ssim_c*ssim_s

        DS2 = (ssim_s - torch.mean(ssim_s, dim=1, keepdim=True)) / reduce_std(ssim_s+1e-8, [1], True)
        DS2 = torch.tanh(DS2)

        CA = F.softmax(lamda * DS2, dim=1)
        if ib == 0:
            CA_batchs = CA
        else:
            CA_batchs = torch.cat([CA_batchs, CA], dim=0)
    return CA_batchs



def save_layer_feature(featuremaps, feature_2, save_path, file_name):
    paddding = nn.ZeroPad2d(5)
    # in_featuremap = paddding(in_featuremap).tanh()/2+0.5
    # out_featuremap = paddding(out_featuremap).tanh()/2+0.5
    # style_feat = paddding(style_feat).tanh()/2+0.5
    featuremaps = featuremaps/2+0.5
    featuremaps = paddding(featuremaps)/2+0.5

    feature_2 = feature_2/2+0.5
    feature_2 = paddding(feature_2)
    
    b, c, h, w = featuremaps.size()
    
    # features = feature_numpy[0]
    featuremaps = torch.mean(featuremaps, dim=1, keepdim=True)
    result = Image.new('RGB', (w*b, h*2+20))
    

    fontsize = 10
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),fontsize)

    # for i in range(c):
    #     in_feature = featuremaps[0:1, i:i+1, :, :]
    #     in_feature_numpy = tensor2im(in_feature)
    #     in_image_pil = Image.fromarray(in_feature_numpy)
    #     result.paste(in_image_pil, box=(w*i, 0))
    for i in range(b):
        in_feature = featuremaps[i:i+1, 0:1, :, :]
        in_feature_numpy = tensor2im(in_feature)
        in_image_pil = Image.fromarray(in_feature_numpy)
        result.paste(in_image_pil, box=(w*i, 0))
        
        in_feature = feature_2[i:i+1, 0:1, :, :]
        in_feature_numpy = tensor2im(in_feature)
        in_image_pil = Image.fromarray(in_feature_numpy)
        result.paste(in_image_pil, box=(w*i, h+10))

    save_name = os.path.join(save_path, file_name+'.jpg')
    result.save(save_name,  quality=100)
    print('save: '+file_name)

def save_feature_value(feature, save_path):
    h, w = feature.size()
    values = []
    for i in range(h):
        value = ''
        for j in range(w):
            value += str(feature[i][j].cpu().item()) +', '
        values.append(value)
    file=open(save_path,'w') 
    for line in values:
        file.write(str(line)+"\n")
    file.close() 