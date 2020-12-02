

import sys, os, yaml,argparse,time
from tqdm import tqdm
from collections import Counter
from munkres import Munkres

import torch

torch.manual_seed(179510)
from resnet import ResNet
from data_aug.dataset_wrapper import DataSetWrapper
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as torchmodels

import numpy as np
np.random.seed(179510)
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score,confusion_matrix
from sklearn import metrics as SKmetrics

from collections import defaultdict

STLCATS = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

def get_data_loaders(data_root,dsname,shuffle=False, batch_size=64,test_only=False,**kwargs):

    args = {}
    if dsname == 'STL10':
        train_dataset = not test_only and  datasets.STL10('{}/{}'.format(data_root,dsname), split='train',transform=transforms.ToTensor())
        test_dataset = datasets.STL10('{}/{}'.format(data_root,dsname), split='test',transform=transforms.ToTensor())
    elif dsname == 'CIFAR10':
        train_dataset = not test_only and datasets.CIFAR10('{}/{}'.format(data_root,dsname), train=True,transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR10('{}/{}'.format(data_root,dsname), train=False,transform=transforms.ToTensor())
    elif dsname.startswith('CIFAR100'):
        train_dataset = not test_only and datasets.CIFAR100('{}/{}'.format(data_root,'CIFAR100'), train=True,transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR100('{}/{}'.format(data_root,'CIFAR100'), train=False,transform=transforms.ToTensor())
    else:
        raise NotImplementedError

    train_loader = not test_only and DataLoader(train_dataset, batch_size=batch_size,
                    num_workers=0, drop_last=False, shuffle=shuffle)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                num_workers=0, drop_last=False, shuffle=shuffle)

    return train_loader, test_loader

class ResNetFeatureExtractor(object):
  def __init__(self, checkpoints_path,data_root,dsname,**kwargs):
    self.dsname = dsname
    self.data_root = data_root
    self.test_only = kwargs.get('test_only',False)
    self.n_classes = kwargs.get('n_classes',-1)
    self.kwargs = kwargs
    config = kwargs['config']
    config.update({'conv1_k3':config['conv1_k3'] or (dsname.startswith('CIFAR') or dsname.startswith('SVHN') )})
    self.config = config
    model = ResNet(**config['model'],conv1_k3 = self.config['conv1_k3'])
    self.model = self._load_resnet_model(model,checkpoints_path)

  def _load_resnet_model(self,model,checkpoints_path):
 
    model.eval()
    state_dict = torch.load(checkpoints_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict,strict=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    return model  

  def _inference(self, loader):
    labels_vector = []
    images = []
    bcnt = 0
    pimaxes = []
    comps = []
    rep_vector =[]
    rep_h_vector = []
    rep_max_vector= []
    rep_cat_vector= []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for batch_x, batch_y in loader:
        images.extend(list(batch_x))
        batch_x = batch_x.to(device)
        labels_vector.extend(batch_y)
        rep,rep_h = self.model(batch_x)
        if 'Mixture' in self.config['model']:
            n_comps = self.config['model']['Mixture']['n_comps']
            pi,mixtures = rep_h
            mixtures = F.normalize(mixtures, dim=2)
            pi = F.softmax(pi,dim=1).unsqueeze(2)
            pimax = torch.max(pi,dim=1)
            pimaxes.extend(pimax[0].squeeze().cpu().detach().numpy())
            comps.extend(pimax[1].squeeze().cpu().detach().numpy())
            pimax = pimax[1]

            ix = pimax.unsqueeze(2).repeat((1,1,self.config['model']['out_dim']))
            mixmax = torch.gather(mixtures,1,ix).squeeze()
            rep_max_vector.extend(mixmax.cpu().detach().numpy())
            
            mixsum = (mixtures*pi).sum(1)
            rep_h_vector.extend(mixsum.cpu().detach().numpy())

            mixcat = mixtures.reshape((mixtures.shape[0],-1))
            rep_cat_vector.extend(mixcat.cpu().detach().numpy())
        else:
            rep_h_vector.extend(rep_h.cpu().detach().numpy())
        rep_vector.extend(rep.cpu().detach().numpy())

    if self.dsname == 'CIFAR100_20' or (self.dsname == 'CIFAR100' and self.n_classes == 20):
        labels_vector = list(map(lambda x: _cifar100_to_cifar20_dict[x.item()],labels_vector))
    ar = lambda x:np.array(x)
    return ar(rep_vector),ar(rep_h_vector),ar(rep_max_vector),ar(rep_cat_vector), ar(labels_vector),images,ar(pimaxes), ar(comps)

  def get_resnet_features(self,reptype):
    self.reptype=reptype
    
    # train_loader, test_loader = get_stl10_data_loaders(download=False)
    cats = self.config['dataset'][self.config['dataset']['dataset']].get('cats',None)
    train_loader, test_loader = get_data_loaders(self.data_root, self.dsname,test_only=self.test_only,cats=cats)

    if self.test_only:
        X_tr=X_tr_fc=X_tr_pimax=X_tr_mxcat= y_tr=images_tr=pimax_tr= comps_tr = None
    else:
        X_tr,X_tr_fc,X_tr_pimax,X_tr_mxcat, y_tr,images_tr,pimax_tr, comps_tr = self._inference(train_loader)
        print("Features shape:\ttrain: {}".format(X_tr.shape))
    X_ts,X_ts_fc,X_ts_pimax,X_ts_mxcat, y_ts,images_ts, pimax_ts,comps_ts = self._inference(test_loader)
    print("Features shape:\ttest: {}\t{}\t{}\t{}".format(X_ts.shape,X_ts_fc.shape,X_ts_pimax.shape,X_ts_mxcat.shape))
    return X_tr,X_tr_fc,X_tr_pimax,X_tr_mxcat, y_tr, X_ts,X_ts_fc,X_ts_pimax,X_ts_mxcat, y_ts, images_tr, images_ts, pimax_tr,pimax_ts, comps_tr, comps_ts

def load_features(data_root,model_path, config_path,n_classes,dsname):
    features = {}
    config = yaml.load(open(config_path, "r"))
    resnet_feature_extractor = ResNetFeatureExtractor(model_path,data_root,dsname,test_only = True,config=config,n_classes = n_classes)
    _,_,_,_,_, X_test,X_test_fc,X_test_fc_mixture,_, y_test, _,_,_,pimax_ts_max,_, comps_ts_max = resnet_feature_extractor.get_resnet_features('fc')
    features.update({'rep':(X_test,),'fc':(X_test_fc,)})
    if 'MIXTURE' in model_path:
        features.update({ 'fcmax':(X_test_fc_mixture,pimax_ts_max,comps_ts_max)})
    features = {'model_name':model_path, 'y_test':y_test,'features':features}
    return features

def make_cost_matrix(c1, c2):
    """
    # Source https://gist.github.com/siolag161/dc6e42b64e1bde1f263b
    """
    uc2 = np.unique(c2)
    uc1 = uc2#np.unique(c1)
    
    l1 = uc1.size
    l2 = uc2.size
    # assert(l1 == l2 and np.all(uc1 == uc2))

    m = np.ones([l1, l2])
    for i in range(l1):
        it_i = np.nonzero(c1 == uc1[i])[0]
        for j in range(l2):
            it_j = np.nonzero(c2 == uc2[j])[0]
            m_ij = np.intersect1d(it_j, it_i)
            m[i,j] =  -m_ij.size
    return m

def unsuper_metrics(X,preds):
    '''
    https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
    '''
    try:
        sil = SKmetrics.silhouette_score(X, preds,sample_size = None, metric='euclidean')
        cal_har = SKmetrics.calinski_harabasz_score(X,preds)
        dav_bou = SKmetrics.davies_bouldin_score(X,preds)
    except ValueError:
        return -100,-100,100
    return sil,cal_har,dav_bou

def clustering_acc(gt,pred):
    cost_matrix = make_cost_matrix(pred, gt)
    m = Munkres()
    indexes = m.compute(cost_matrix)
    mapper = { old: new for (old, new) in indexes }
    new_labels = np.array([ mapper[i] for i in pred ])
    num_labels = len(np.unique(gt))
    new_cm = confusion_matrix(gt, new_labels, labels=range(num_labels))
    _acc = np.trace(new_cm, dtype=float) / np.sum(new_cm)
    return _acc

def cluster_metrics(true, preds,n_cls,no_acc=False):
    nmi = normalized_mutual_info_score(true, preds)
    ari = adjusted_rand_score(true, preds)
    acc = (no_acc and -1) or clustering_acc(true, np.array(preds))
    return nmi,ari,acc

def cat_avg_X(X,y,cats):

    if isinstance(cats,int):
        pred_cls_counts = sorted(Counter(y).items(),key = lambda x: x[1],reverse=True)
        cats = [x[0] for x in pred_cls_counts[:cats]]

    means = np.empty((len(cats),X.shape[1]))
    for idx,c in enumerate(cats):
        sampsc = np.sum((y==c).astype(int))
        if sampsc == 0:
            _min = X.min()
            _max = X.max()
            rnd = np.random.rand(X.shape[1])
            means[c] = _min + (rnd*(_max-_min))
        else:
            repsc = X[y==c]
            means[idx] = repsc.mean(axis=0)
    return means

def cluster(reps,n_classes):
    kmclusters = {}
    name = reps['model_name']

    features = reps['features']    
    max_iter = 10
    for normalize in [False,True]:
        for ftype in features:
            if ftype not in kmclusters:
                kmclusters.update({ftype:{}})
            X = features[ftype][0]
            norm_prefix =''
            if normalize:
                X= X/np.linalg.norm(X,axis=1,keepdims=True)
                norm_prefix = 'norm_'
            clttye = norm_prefix + 'KMeans w/ random init'
            if clttye not in kmclusters[ftype]:
                time_start = time.time()
                kmeans_sk = KMeans(n_clusters=n_classes,n_init= 50,max_iter=max_iter, random_state=0).fit(X)
                time_end = time.time()
                total_time = time_end-time_start
                preds = kmeans_sk.predict(X)
                kmclusters[ftype].update({clttye:(preds,unsuper_metrics(X,preds),kmeans_sk,total_time)})

            clttye= norm_prefix + 'KMeans w/ mean component init'
            if 'MIXTURE' in name and clttye not in kmclusters[ftype]:
                y = features['fcmax'][-1]
                cat_means = cat_avg_X(X,y,list(range(n_classes)))
                time_start = time.time()
                kmeans_sk = KMeans(n_clusters=n_classes,init=cat_means,n_init=1,max_iter=max_iter, random_state=0).fit(X)
                time_end = time.time()
                total_time = time_end-time_start
                preds = kmeans_sk.predict(X)
                kmclusters[ftype].update({clttye:(preds,unsuper_metrics(X,preds),kmeans_sk,total_time)})
            if 'MIXTURE' in name and 'MIXTURE_{}'.format(n_classes) not in name:
                clttye= norm_prefix + 'KMeans_FRQBASED_{}'.format(ftype)
                if 'MIXTURE' in name and clttye not in kmclusters[ftype]:
                    y = features['fcmax'][-1]
                    cat_means = cat_avg_X(X,y,n_classes)
                    time_start = time.time()
                    kmeans_sk = KMeans(n_clusters=n_classes,init=cat_means,n_init=1,max_iter=max_iter, random_state=0).fit(X)
                    time_end = time.time()
                    total_time = time_end-time_start
                    preds = kmeans_sk.predict(X)
                    kmclusters[ftype].update({clttye:(preds,unsuper_metrics(X,preds),kmeans_sk,total_time)})
            
            if not normalize and ftype == 'fcmax' and 'Mixture' not in kmclusters[ftype]:
                preds =features[ftype][-1]
                kmclusters[ftype].update({'Mixture':(preds,unsuper_metrics(X,preds))})
          
    kmclusters = {'model_name':name, 'clusters':kmclusters}
    return kmclusters

def eval_clusters(y_test,kmclusters,n_classes):
    metrics = {}
    name = kmclusters['model_name']
    kmclusters = kmclusters['clusters']
    for ftype in kmclusters:
        if ftype not in metrics:
            metrics.update({ftype:{ }})
        y = y_test
        for cluster_type in sorted(kmclusters[ftype]):
            if cluster_type not in metrics[ftype]:
                metrics[ftype].update({cluster_type:{}})
                clts = kmclusters[ftype][cluster_type][0]
                no_acc = cluster_type == 'Mixture' and 'MIXTURE_{}'.format(n_classes) not in name
                nmi,ari,acc = cluster_metrics(y,clts,n_classes,no_acc=no_acc)
                metrics[ftype][cluster_type].update({'nmi':nmi})
                metrics[ftype][cluster_type].update({'ari':ari})
                metrics[ftype][cluster_type].update({'acc':acc})

                sil,cal_har,dav_bou = kmclusters[ftype][cluster_type][1]

                metrics[ftype][cluster_type].update({'sil':sil})
                metrics[ftype][cluster_type].update({'cal_har':cal_har})
                metrics[ftype][cluster_type].update({'dav_bou':dav_bou})

                cnt = Counter(clts)
                p = np.array([cnt[c] for c in range(n_classes)])
                p = p/p.sum()
                ent = entropy(p)
                metrics[ftype][cluster_type].update({'ent':ent})

                if cluster_type != 'Mixture':
                    inertia = kmclusters[ftype][cluster_type][2].inertia_
                    metrics[ftype][cluster_type].update({'inertia':inertia})
    return metrics

def entropy(p,dohist=False):
    p = np.array(p)
    if dohist:
        p,_ = np.histogram(p,bins =100, normed=True, density=True)
        p = p/p.sum()
    if np.any(p==0):
        p += 1e-10
        p = p/p.sum()
    en = np.sum(p*np.log2(p))*-1
    return en

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # basic experiment setting
    
    parser.add_argument('data_root',type=str, default=None)
    parser.add_argument('dsname',type=str, default=None)
    parser.add_argument('model_path',type=str, default=None)
    parser.add_argument('config_path',type=str, default=None)
    parser.add_argument('n_classes',type=int, default=None)

    args = parser.parse_args(sys.argv[1:])
    for k,v in sorted(vars(args).items()):
        print('{}: {}'.format(str(k).ljust(20),v ) )

    features = load_features(args.data_root,args.model_path, args.config_path, args.n_classes,args.dsname)
    kmclusters = cluster(features, args.n_classes)
    metrics = eval_clusters(features['y_test'],kmclusters, args.n_classes)

    for ftype in metrics:
        print('#'*100);print('Feature type:', ftype)
        for cltconf in metrics[ftype]:
            print('-'*100)
            print('Cluster config:', cltconf)
            print('Metrc'.ljust(20), 'Score'.ljust(20))
            for m in metrics[ftype][cltconf]:
                print(str(m).ljust(20),  str(round(metrics[ftype][cltconf][m],2)).ljust(20) )