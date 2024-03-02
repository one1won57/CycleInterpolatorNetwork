import torch
import numpy as np

def real_imag(kspace, coilN):
    kspace_real = kspace[:,:coilN,:,:]
    kspace_imag = kspace[:,coilN:,:,:]
    res = kspace_real +1j*kspace_imag
    return res 

def RSSq(img):
    return (torch.sum(abs(img)**2,1)**(0.5)).unsqueeze(0)

def mfft1(real_imag_kspace, dim = 2):
    res = torch.fft.ifftshift(real_imag_kspace,dim=dim)
    res = torch.fft.ifft(res,dim=dim)
    res = torch.fft.ifftshift(res,dim=dim)
    return res

def mfft2(real_imag_kspace,dimx =2,dimy=3 ):
    fft1_kspace = mfft1(real_imag_kspace,dim=dimx)
    fft2_kspace = mfft1(fft1_kspace,dim=dimy)
    return fft2_kspace

def RSSq_np(img):
    return np.expand_dims((np.sum(np.abs(img)**2,1)**(0.5)),axis=0)


def mfft1_np(real_imag_kspace, axis = 2):
    res = np.fft.ifftshift(real_imag_kspace,axes=axis)
    res = np.fft.ifft(res,axis=axis)
    res = np.fft.ifftshift(res,axes=axis)
    return res

def mfft2_np(real_imag_kspace,dimx =2,dimy=3 ):
    fft1_kspace = mfft1_np(real_imag_kspace,axis=dimx)
    fft2_kspace = mfft1_np(fft1_kspace,axis=dimy)
    return fft2_kspace

