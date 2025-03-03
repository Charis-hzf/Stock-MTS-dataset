import pickle
import numpy as np
import torch
from .utils import _init_data_loader
from torch.utils.data import TensorDataset,DataLoader

def dataloaders_select(dataset,model_name):
    dataset=dataset.upper()
    model_name=model_name.upper()

    if dataset in ('NASDAQ','NYSE'):
        return load_masked_data
    elif dataset in ('SP500'):
        return load_sp
    
    else:
        raise NotImplementedError


    
def load_masked_data(dataset,batch_size,shuffle_train,n_jobs):
    dataset=dataset.upper()
    with open(f'/home/cseadmin/hzf/TorchStock/data/{dataset}/eod_data.pkl','rb')as f:
        eod_data=pickle.load(f)
    with open(f'/home/cseadmin/hzf/TorchStock/data/{dataset}/gt_data.pkl','rb')as f:
        gt_data=pickle.load(f)
    with open(f'/home/cseadmin/hzf/TorchStock/data/{dataset}/price_data.pkl','rb')as f:
        price_data=pickle.load(f)
    with open(f'/home/cseadmin/hzf/TorchStock/data/{dataset}/mask_data.pkl','rb')as f:
        mask_data=pickle.load(f)
    steps=1
    window=16
    x=[]
    y=[]
    #1245-window-steps+1
    valid_index=756-window+steps-1
    test_index=1008-window+steps-1
    for idx in range(1245-window-steps+1):
        # 16-days feature
        x.append(eod_data[:,idx:idx+window,:])
        # 17-days mask + 1 day base + 1 day gt
        mask_sample=mask_data[:,idx:idx+window+1]
        price_sample=np.expand_dims(price_data[:,idx+window-1],axis=1)
        gt_sample=np.expand_dims(gt_data[:,idx+window+steps-1],axis=1)
        mask_sample=np.min(mask_sample,axis=1)
        final_mask=np.expand_dims(mask_sample, axis=1)
        y.append(np.concatenate((gt_sample,price_sample,final_mask),axis=1))
    x=np.array(x)
    y=np.array(y)
    x_train=x[:valid_index]
    x_valid=x[valid_index:test_index]
    x_test=x[test_index:]
    y_train=y[:valid_index]
    y_valid=y[valid_index:test_index]
    y_test=y[test_index:]

    trainset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    valset = TensorDataset(torch.FloatTensor(x_valid), torch.FloatTensor(y_valid))
    testset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
    trainset_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle_train)
    valset_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    testset_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainset_loader,valset_loader,testset_loader
    
def load_sp(dataset, batch_size, shuffle_train, n_jobs):
    dataset=dataset.upper()
    data = np.load(f'/home/cseadmin/hzf/TorchStock/data/{dataset}/{dataset}.npy')[:,915:,:]
    price_data = data[:, :, -1]
    mask_data=np.ones((data.shape[0], data.shape[1]))
    eod_data = data
    gt_data = np.zeros((data.shape[0], data.shape[1]))
    steps=1
    window=16
    for ticket in range(0, data.shape[0]):
        for row in range(1, data.shape[1]):
            gt_data[ticket][row] = (data[ticket][row][-1] - data[ticket][row - steps][-1]) / \
                                data[ticket][row - steps][-1] 
            
    x=[]
    y=[]
    valid_index=1006-window+steps-1
    test_index=1259-window+steps-1
    for idx in range(1611-window-steps+1):
        # 16-days feature
        x.append(eod_data[:,idx:idx+window,:])
        # 17-days mask + 1 day base + 1 day gt
        mask_sample=mask_data[:,idx:idx+window+1]
        price_sample=np.expand_dims(price_data[:,idx+window-1],axis=1)
        gt_sample=np.expand_dims(gt_data[:,idx+window+steps-1],axis=1)
        mask_sample=np.min(mask_sample,axis=1)
        final_mask=np.expand_dims(mask_sample, axis=1)
        y.append(np.concatenate((gt_sample,price_sample,final_mask),axis=1))
    x=np.array(x)
    y=np.array(y)
    x_train=x[:valid_index]
    x_valid=x[valid_index:test_index]
    x_test=x[test_index:]
    y_train=y[:valid_index]
    y_valid=y[valid_index:test_index]
    y_test=y[test_index:]

    trainset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    valset = TensorDataset(torch.FloatTensor(x_valid), torch.FloatTensor(y_valid))
    testset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
    trainset_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle_train)
    valset_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    testset_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainset_loader,valset_loader,testset_loader
