

import torch
import torch.nn as nn
from torchinfo import summary
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import scipy.io as sio
import time
import os

from nnModel import network
from Utils import real_imag, mfft2, save_rssq

# Parameter setting    
acsN, acc_rate, slIdx, epochs = 8, 2, 0, 1000
inputData = "[file_name.mat]"
result, gpu, batch,  lr, lossLev = "save_result", 0, 1,  3e-3, 1000000000

#### Input/Output Data ####
dir_data = "home/user/"
input_variable_name = 'pEPI_full_kspace'
recon_variable_name = 'kspace_recon'

## Preparing Saving Folders
if not os.path.exists(result):
    os.makedirs(result)
    

############################ Setting GPU  ###########################
## CPU thread
torch.set_num_threads(4)

# GPU
GPU_NUM = gpu # GPU number

device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print('##################################')
print ('Current cuda device ', torch.cuda.current_device()) # check
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
print('##################################')


kspace_1subject = sio.loadmat(dir_data +  inputData)
kspace_1subject = kspace_1subject[input_variable_name] 
kspace_full = kspace_1subject[slIdx:slIdx+1:,:,:].transpose(0,1,3,2)

print('loaded kspace      : ' + str(kspace_full.shape)) # (# of slice, no_ch*2, ky, kx)
print('kspace dataset     : %s' % (inputData,))

# prepariing kspace data (referenced from RAKI code)
normalize = 1/np.max(abs(kspace_full[:]))
kspace_full = np.multiply(kspace_full,normalize)   
kspace = np.zeros(kspace_full.shape)
kspace[:,:,::acc_rate,:] = kspace_full[:,:,::acc_rate,:]

[tmp, no_ch, m1,n1] = np.shape(kspace)
coilN = int(no_ch/2)

ky = np.transpose(np.int32([(range(1,m1+1))]))                  
kx = np.int32([(range(1,n1+1))])

mask = np.squeeze(np.sum(np.sum(np.abs(kspace),3),1))>0 
picks = np.where(mask == 1);                                  
kspace = kspace[:,:,np.int32(picks[0][0]):n1+1,:]
kspace_all = kspace_full[:,:,np.int32(picks[0][0]):n1+1,:]  
kspace_NEVER_TOUCH = np.copy(kspace_all) #fullsampled kspace, normalized kspace

mask = np.squeeze(np.sum(np.sum(np.abs(kspace),3),1))>0;  
picks = np.where(mask == 1);                                  
d_picks = np.diff(picks,1)  
indic = np.where(d_picks == 1)

mask_x = np.squeeze(np.sum(np.sum(np.abs(kspace),1),0))>0
picks_x = np.where(mask_x == 1)
x_start = picks_x[0][0]
x_end = picks_x[0][-1]

ACS = kspace_NEVER_TOUCH[:,:,int(n1/2-acsN/2):int(n1/2+acsN/2),:]
[tmp, full_dim_Z, full_dim_Y, full_dim_X] = np.shape(kspace_NEVER_TOUCH)
[tmp, ACS_dim_Z, ACS_dim_Y, ACS_dim_X] = np.shape(ACS)
ACS_re = ACS

acc_rate = d_picks[0][0]
no_channels = ACS_dim_Z

#### Network Parameters ####
kernel_x_1 = 5
kernel_y_1 = 2

kernel_x_2 = 1
kernel_y_2 = 1

kernel_last_x = 3
kernel_last_y = 2

layer1_channels = 32 
layer2_channels = 8

target_x_start = np.int32(np.ceil(kernel_x_1/2) + np.floor(kernel_x_2/2) + np.floor(kernel_last_x/2) -1); 
target_x_end = np.int32(ACS_dim_X - target_x_start -1); 

target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * acc_rate     
target_y_end = ACS_dim_Y  - np.int32((np.floor(kernel_y_1/2) + np.floor(kernel_y_2/2) + np.floor(kernel_last_y/2))) * acc_rate -1

target_dim_X = target_x_end - target_x_start + 1
target_dim_Y = target_y_end - target_y_start + 1
target_dim_Z = acc_rate - 1

target_y_end_cyc = full_dim_Y  - np.int32((np.floor(kernel_y_1/2) + np.floor(kernel_y_2/2) + np.floor(kernel_last_y/2))) * acc_rate -1
target_dim_Y_cyc = target_y_end_cyc- target_y_start + 1

X_test = torch.FloatTensor(kspace_all) # kspace_full_sampled
testset = TensorDataset(X_test)
testloader= DataLoader(testset,batch_size=batch,shuffle=False)
test_features = next(iter(testloader))
print(f"test_X batch shape(full): {test_features[0].size()} ")
print('coilN, acsN: ', coilN, acsN)

PATH_i = []
pEPInn_i = []
optimizer_i = []

for i in range(no_ch):
    pEPInn_i.append(network(acc_rate,no_ch).to(device))
    optimizer_ = torch.optim.Adam(pEPInn_i[-1].parameters(), lr=lr)
    optimizer_i.append(optimizer_)

print('Model Summary')
summary(pEPInn_i[0], input_size = test_features[0].size())


'''
# for debug

epoch = 0
data = next(iter(testloader))
bs,i, loop = 0,0, 0
batch_loss_c = 0
losses = []
batch_losses = []
'''

 
for i in range(no_ch):
    pEPInn_i[i].train()

'''
Cycle Interpolator Optimizing
'''
losses = []
print('### Start ###')
time_ALL_start = time.time()
for epoch in range(epochs):
    batch_loss_c = 0
    batch_losses = []
    
    for i,data in enumerate(testloader):
        
        Y_batch_full_cyc = data[0] # full size kspace
        Y_batch_ACS_cyc = Y_batch_full_cyc[:,:,int(m1/2-acsN/2):int(m1/2+acsN/2),:] #ACS lines
        Y_batch_ACS_cyc = Y_batch_ACS_cyc.to(device)
        
        X_batch_full_even_cyc = torch.zeros(Y_batch_full_cyc.size(), device=device)
        X_batch_full_even_cyc[:,:,::acc_rate,:] = Y_batch_full_cyc[:,:,::acc_rate,:].to(device) # undersampled whole data
           
        
        est_kspace_ACS_cyc_i = [pEPInn_i[j](Y_batch_ACS_cyc) for j in range(no_ch)]
        est_kspace_full_even_odd1_cyc_i = [pEPInn_i[j](X_batch_full_even_cyc) for j in range(no_ch)]
        target = torch.zeros([1,no_ch*(acc_rate-1), target_dim_Y,target_dim_X], device=device)
        

        for ind_acc in range(acc_rate-1):
            target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * acc_rate + ind_acc + 1 
            target_y_end = ACS_dim_Y  - np.int32((np.floor(kernel_y_1/2) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2))) * acc_rate + ind_acc
            target[:,ind_acc::acc_rate-1,:,:] = Y_batch_ACS_cyc[:, :,target_y_start:target_y_end +1,target_x_start:target_x_end + 1]

        recon_kspace_full_even_odd1_cyc = torch.zeros(Y_batch_full_cyc.size(), device=device)
        recon_kspace_full_even_odd1_cyc[:,:,::acc_rate,:] = X_batch_full_even_cyc[:,:,::acc_rate,:]
        [tmp, dim_kspaceUnd_Z,dim_kspaceUnd_Y,dim_kspaceUnd_X] = np.shape(recon_kspace_full_even_odd1_cyc)
        target_x_end_kspace = dim_kspaceUnd_X - target_x_start

        for ind_acc in range(acc_rate-1):
            target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + np.int32((np.ceil(kernel_y_2/2)-1)) + np.int32(np.ceil(kernel_last_y/2)-1)) * acc_rate + ind_acc + 1             
            target_y_end_kspace = dim_kspaceUnd_Y - np.int32((np.floor(kernel_y_1/2)) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2)) * acc_rate + ind_acc
            for j in range(no_ch):
                recon_kspace_full_even_odd1_cyc[:,j,target_y_start:target_y_end_kspace+1:acc_rate,target_x_start:target_x_end_kspace]  = est_kspace_full_even_odd1_cyc_i[j][:,ind_acc,::acc_rate,:].detach()
        


        '''
        cycle
        '''

        recon_kspace_full_odd1_cyc = torch.clone(recon_kspace_full_even_odd1_cyc)
        recon_kspace_full_odd1_cyc[:,:,::acc_rate,:] = 0

        est_kspace_full_even2_odd1_cyc_i = [pEPInn_i[j](recon_kspace_full_odd1_cyc) for j in range(no_ch)]
        target_cyc = torch.zeros([1,no_ch*(acc_rate-1), target_dim_Y_cyc,target_dim_X], device=device)



        recon_kspace_full_even2_odd1_cyc = torch.zeros(Y_batch_full_cyc.size(), device=device)
        recon_kspace_full_even2_odd1_cyc[:,:,1::acc_rate,:] = recon_kspace_full_odd1_cyc[:,:,1::acc_rate,:]
        [tmp, dim_kspaceUnd_Z_cyc, dim_kspaceUnd_Y_cyc, dim_kspaceUnd_X_cyc] = np.shape(recon_kspace_full_even2_odd1_cyc)
        target_x_end_kspace_cyc = dim_kspaceUnd_X_cyc - target_x_start                
        
        for ind_acc in range(acc_rate-1):
            target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * acc_rate + ind_acc + 1 
            target_y_end = dim_kspaceUnd_Y_cyc  - np.int32((np.floor(kernel_y_1/2) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2))) * acc_rate + ind_acc
            target_cyc[:,ind_acc::acc_rate-1,:,:] = X_batch_full_even_cyc[:, :,target_y_start:target_y_end +1,target_x_start:target_x_end + 1]

        

        ind_acc = 0
        target_y_start_cyc = np.int32((np.ceil(kernel_y_1/2)-1) + np.int32((np.ceil(kernel_y_2/2)-1)) + np.int32(np.ceil(kernel_last_y/2)-1)) * acc_rate + ind_acc +1             
        target_y_end_kspace_cyc = dim_kspaceUnd_Y_cyc - np.int32((np.floor(kernel_y_1/2)) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2)) * acc_rate + ind_acc
    
        
        for j in range(no_ch):
            recon_kspace_full_even2_odd1_cyc[:,j,acc_rate:target_y_end_kspace_cyc+1:acc_rate,target_x_start:target_x_end_kspace_cyc] \
                = est_kspace_full_even2_odd1_cyc_i[j][:,ind_acc,acc_rate-1::acc_rate,:].detach()
    

        '''
        cyc loss
        '''
        mse_loss = nn.MSELoss()

        cycloss_i = [0]*no_ch
        lossCycACS_i = [0]*no_ch
        totLoss_i = [0]*no_ch
        

        for ind_acc in range(acc_rate-1):
            for j in range(no_ch):
                cycloss_i[j] += mse_loss(target_cyc[:,ind_acc+(acc_rate-1)*j,(acc_rate-1)-ind_acc::acc_rate,:],    est_kspace_full_even2_odd1_cyc_i[j][:,ind_acc,(acc_rate-1)-ind_acc::acc_rate,:])
                lossCycACS_i[j] += mse_loss(target[:,ind_acc+(acc_rate-1)*j,:,:], est_kspace_ACS_cyc_i[j][:,ind_acc,:,:])
                totLoss_i[j]+=cycloss_i[j]+lossCycACS_i[j]

        for j in range(no_ch): 
            optimizer_i[j].zero_grad()
            totLoss_i[j].backward()        
            optimizer_i[j].step()  
            
        batch_loss_c += totLoss_i[0]
        batch_losses.append(torch.stack(totLoss_i))
        
        if epoch == 0:
            print(f"Epoch: {epoch+1:05d} Train Loss = {totLoss_i[0]}")
    losses.append(torch.mean(torch.stack(batch_losses,0),0))
                
print(f"Epoch: {epoch+1:05d} Train Loss = {totLoss_i[0]}")

time_ALL_end = time.time()
print("time processed: " +str(time_ALL_end-time_ALL_start))
save_losses = torch.stack(losses,1).detach().cpu().numpy()

## SAVE IMAGE
recon_IMG_full_even_odd1_cyc = mfft2(real_imag(recon_kspace_full_even_odd1_cyc, coilN))
rssq_recon_even_odd1_cyc = torch.squeeze(torch.sum(abs(recon_IMG_full_even_odd1_cyc)**2,1)**(0.5)).cpu().detach().numpy()

Y_batch_full_IMG_cyc =mfft2(real_imag(Y_batch_full_cyc,coilN))
rssq_GT_cyc = torch.squeeze(torch.sum(abs(Y_batch_full_IMG_cyc)**2,1)**(0.5)).cpu().detach().numpy()

save_rssq(rssq_GT_cyc, name='rssq_GT',perc=100, save_img_path=result)
save_rssq(rssq_recon_even_odd1_cyc, name='rssq_recon_cyc',perc=100, save_img_path=result)
