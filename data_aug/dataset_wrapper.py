'''
Adapted from https://github.com/sthalles/SimCLR/tree/master/data_aug
'''
import os, time
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import datasets

class DataSetWrapper(object):

    def __init__(self,dsname, batch_size, num_workers, valid_size, 
                    input_shape, s,minscale,dataroot,**kwargs):
        self.dsname = dsname
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataroot = dataroot
        self.valid_size = valid_size
        self.s = s
        self.minscale = minscale
        self.maxscale = kwargs.get('maxscale',1)
        self.train_sampler = None 
        self.num_trans = 2
        self.input_shape = eval(input_shape)
        normtranses = { 'CIFAR10':transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        'CIFAR100':transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                        'STL10':transforms.Normalize((0., 0., 0.), (1., 1., 1.))
                    }
        self.normtrans = normtranses[self.dsname.split('_')[0]]
        self.kwargs = kwargs

    
    def get_data_loaders(self,resmaple = False):

        color_conf = {'no_jitter':self.kwargs.get('no_jitter',False),'no_gray_scale':self.kwargs.get('no_gray_scale',False)} 
        data_augment = self._transform_pipeline(self.input_shape[0],self.s,self.minscale,self.maxscale,self.normtrans,**color_conf)
        dsargs={'transform':SimCLRDataTransform(data_augment,self.num_trans), 'download':True}

        dsroot='{}/{}'.format(self.dataroot,self.dsname.split('_')[0])
            
        if not os.path.exists(dsroot):
            os.mkdir(dsroot)
       
        def stltraintest():
            tr = datasets.STL10(dsroot, split='train', **dsargs)
            ts = datasets.STL10(dsroot, split='test', **dsargs)
            concat = torch.utils.data.ConcatDataset((tr,ts))
            return concat
        def cifartraintest():
            dsfn = datasets.CIFAR100 if self.dsname.startswith('CIFAR100') else datasets.CIFAR10
            tr = dsfn(dsroot, train=True, **dsargs)
            ts = dsfn(dsroot, train=False, **dsargs)
            concat = torch.utils.data.ConcatDataset((tr,ts))
            return concat
        def svhntraintest():
            tr = datasets.SVHN(dsroot, split='train', **dsargs)
            ts = datasets.SVHN(dsroot, split='test', **dsargs)
            concat = torch.utils.data.ConcatDataset((tr,ts))
            return concat
        # print('pytorch dataset')
        __cat = lambda x :torch.utils.data.ConcatDataset((x,))
        dataset_fns = { 'STL10':lambda: __cat(datasets.STL10(dsroot, split='train+unlabeled', **dsargs)),
                        'STL10_Train':lambda: __cat(datasets.STL10(dsroot, split='train', **dsargs)),
                        'STL10_Test':lambda: __cat(datasets.STL10(dsroot, split='test', **dsargs)),
                        'STL10_TrainTest': stltraintest, 
                        'CIFAR10':lambda: __cat(datasets.CIFAR10(dsroot, train=True, **dsargs)),
                        'CIFAR10_Test':lambda: __cat(datasets.CIFAR10(dsroot, train=False, **dsargs)),
                        'CIFAR10_TrainTest': cifartraintest,
                        'CIFAR100':lambda: __cat(datasets.CIFAR100(dsroot, train=True, **dsargs)),
                        'CIFAR100_Test':lambda: __cat(datasets.CIFAR100(dsroot, train=False, **dsargs)),
                        'CIFAR100_TrainTest': cifartraintest,
                        'CIFAR100_20':lambda: __cat(datasets.CIFAR100(dsroot, train=True, **dsargs)),
                        'CIFAR100_20_Test':lambda: __cat(datasets.CIFAR100(dsroot, train=False, **dsargs)),
                        'CIFAR100_20_TrainTest': cifartraintest,
                        }
        train_dataset = dataset_fns[self.dsname]()
        
        # obtain training indices that will be used for validation
        if self.train_sampler is None or resmaple:
            num_train = len(train_dataset)
            print('num_train:',num_train)
            indices = list(range(num_train))
            np.random.shuffle(indices)
            if self.valid_size == -1:
                train_idx, valid_idx = indices, indices
            else:
                split = int(np.floor(self.valid_size * num_train))
                train_idx, valid_idx = indices[split:], indices[:split]
            print('len(train_idx):',len(train_idx),'len(valid_idx):',len(valid_idx))
            # define samplers for obtaining training and validation batches
            self.train_sampler = SubsetRandomSampler(train_idx)
            self.valid_sampler = SubsetRandomSampler(valid_idx)
        
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        print('len(train_loader):',len(train_loader),'len(valid_loader):',len(valid_loader))

        return train_loader, valid_loader

    def _transform_pipeline(self,input_shape,s,minscale,maxscale,normtrans,**kwargs):
        
        # get a set of data augmentation transformations as described in the SimCLR paper.
        _trans = [transforms.RandomResizedCrop(size=input_shape,scale=(minscale,maxscale)),
                                            transforms.RandomHorizontalFlip()]
        if not kwargs.get('no_jitter',False):
            color_jitter = transforms.ColorJitter(0.8 *s, 0.8 * s, 0.8 * s, 0.2 * s)
            _trans.append(transforms.RandomApply([color_jitter], p=0.8))
        if not kwargs.get('no_gray_scale',False):
            gray_scale = 0.2    
            _trans.append(transforms.RandomGrayscale(p=gray_scale))
        _trans.extend([GaussianBlur(kernel_size=int(0.1 * input_shape)),
                                            transforms.ToTensor(),
                                            normtrans])
        data_transforms = transforms.Compose(_trans)
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                sampler=self.train_sampler,
                                num_workers=self.num_workers,
                                drop_last=True, shuffle=False,
                                pin_memory=True,prefetch_factor=2)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                sampler=self.valid_sampler,
                                num_workers=self.num_workers, 
                                drop_last=True,
                                pin_memory=True,prefetch_factor=2)
        return train_loader, valid_loader


class SimCLRDataTransform(object):
    def __init__(self, transform,num_trans):
        self.transform = transform
        self.num_trans =num_trans
        
        self.trans = [transform]*self.num_trans or transform

    def __call__(self, sample):

        ret = list(map(lambda trans: trans(sample), self.trans))

        return ret


