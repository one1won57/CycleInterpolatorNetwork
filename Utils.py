import torch
import numpy as np
import matplotlib.pyplot as plt

def real_imag(kspace, coilN):
    kspace_real = kspace[:,:coilN,:,:]
    kspace_imag = kspace[:,coilN:,:,:]
    res = kspace_real +1j*kspace_imag
    return res 

def mfft1(real_imag_kspace, dim = 2):
    res = torch.fft.ifftshift(real_imag_kspace,dim=dim)
    res = torch.fft.ifft(res,dim=dim)
    res = torch.fft.ifftshift(res,dim=dim)
    return res

def mfft2(real_imag_kspace,dimx =2,dimy=3 ):
    fft1_kspace = mfft1(real_imag_kspace,dim=dimx)
    fft2_kspace = mfft1(fft1_kspace,dim=dimy)
    return fft2_kspace

def save_rssq(img, perc, save_img_path, name = 'rssq', cmap = 'gray', show = False):
    img = np.fliplr(np.flipud(img.transpose(1,0)))
    img = img/np.percentile(img,perc)
    plt.imsave(save_img_path + '/'+name +'.jpg', img, vmin =0,vmax = 1, cmap = cmap)
    if show == True:
        plt.imshow(img, cmap = cmap, vmin=0,vmax=1)
    return None
