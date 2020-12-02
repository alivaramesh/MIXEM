'''
Adapted form https://github.com/sthalles/SimCLR/tree/master/feature_eval
'''
import sys, os,argparse, pickle,time
import torch
torch.manual_seed(179510)
import numpy as np
np.random.seed(179510)
from resnet import ResNet
import yaml
from sklearn import preprocessing
import importlib.util

from tqdm import tqdm
import torch.nn as nn

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

_cifar100_to_cifar20_dict = {0: 4,1: 1,2: 14,3: 8,4: 0,5: 6,6: 7,7: 7,8: 18,9: 3,10: 3,11: 14,12: 9,13: 18,14: 7,15: 11,16: 3,17: 9,18: 7,19: 11,20: 6,21: 11,22: 5,23: 10,24: 7,25: 6,26: 13,27: 15,28: 3,29: 15,30: 0,31: 11,32: 1,33: 10,34: 12,35: 14,36: 16,37: 9,38: 11,39: 5,40: 5,41: 19,42: 8,43: 8,44: 15,45: 13,46: 14,47: 17,48: 18,49: 10,50: 16,51: 4,52: 17,53: 4,54: 2,55: 0,56: 17,57: 4,58: 18,59: 17,60: 10,61: 3,62: 2,63: 12,64: 12,65: 16,66: 12,67: 1,68: 9,69: 19,70: 2,71: 10,72: 0,73: 1,74: 16,75: 12,76: 9,77: 13,78: 15,79: 13,80: 16,81: 19,82: 2,83: 4,84: 6,85: 19,86: 5,87: 5,88: 8,89: 19,90: 18,91: 1,92: 2,93: 15,94: 6,95: 0,96: 17,97: 8,98: 14,99: 13}

def _load_resnet_model(model_path):
  # Load the neural net module

  model = ResNet(**config["model"],conv1_k3 = config['conv1_k3']).to(device)

  state_dict = torch.load(model_path, map_location=torch.device('cpu'))
  model.load_state_dict(state_dict)
  model = model.to(device)
  return model

def get_data_loaders(dsname,shuffle=False, cats='',pickle_path=None,**kwargs):

  global data_root
  global batch_size
  insize = kwargs.get('insize',96)
  print('>>> insize:',insize)
  args = {}
  print('>>> dsname:',dsname)
  if dsname == 'STL10':
    trans = transforms.ToTensor()
    if insize != 96:
      trans = transforms.Compose([transforms.Resize((insize,insize)),transforms.ToTensor()])
    train_dataset = datasets.STL10('{}/{}'.format(data_root,dsname), split='train',transform=trans)
    test_dataset = datasets.STL10('{}/{}'.format(data_root,dsname), split='test',transform=trans)
  elif dsname == 'CIFAR10':
    train_dataset = datasets.CIFAR10('{}/{}'.format(data_root,dsname), train=True,transform=transforms.ToTensor())
    test_dataset = datasets.CIFAR10('{}/{}'.format(data_root,dsname), train=False,transform=transforms.ToTensor())
  elif dsname == 'CIFAR100':
    train_dataset = datasets.CIFAR100('{}/{}'.format(data_root,dsname), train=True,transform=transforms.ToTensor())
    test_dataset = datasets.CIFAR100('{}/{}'.format(data_root,dsname), train=False,transform=transforms.ToTensor())
  print('Creating data loaders ...')
  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                  num_workers=2, drop_last=False, shuffle=shuffle)

  test_loader = DataLoader(test_dataset, batch_size=batch_size,
                num_workers=2, drop_last=False, shuffle=shuffle)
  print('Data loaders created.')
  return train_loader,len(train_dataset), test_loader, len(test_dataset)


class LogisticRegression(nn.Module):
  
  def __init__(self, n_features, n_classes):
    super(LogisticRegression, self).__init__()
    self.model = nn.Linear(n_features, n_classes)

  def forward(self, x):
    return self.model(x)

class ResNetFeatureExtractor(object):
  def __init__(self, model_path,dsname,**kwargs):
    self.dsname = dsname
    self.insize = kwargs.get('insize',96)
    self.cats = kwargs.get('cats','')
    self.pickle_path = kwargs.get('pickle_path',None)
    self.model = kwargs.get('model')
    if self.model is None:
      self.model = _load_resnet_model(model_path)
    self.model.eval()

  def _inference(self, loader,len_loader):
    feature_vector = []
    labels_vector = []
    time_fw = 0
    time_store = 0
    time_cpu=0
    time_np =0
    n_iter = 0
    # tmp_batch = next(iter(loader))[0].to(device)
    # BS = tmp_batch.shape[0]
    
    feature_vector = None
    with torch.no_grad():
      
      for batch_x, batch_y in tqdm(loader):
        n_iter += 1
        
        batch_x = batch_x.to(device)
        labels_vector.extend(batch_y)
        s = time.time()
        features = self.model(batch_x)[0]
        time_fw += (time.time()-s)

        s = time.time()
        features = features.cpu()
        time_cpu += (time.time()-s)

        s = time.time()
        features = features.numpy()
        time_np += (time.time()-s)

        if feature_vector is None:
          BS = features.shape[0]
          feature_vector = np.empty((len_loader,features.shape[1])).astype(np.float32)
          print('feature_vector.shape:',feature_vector.shape)
        
        s= time.time()
        # feature_vector.extend(features.cpu().numpy())
        six = (n_iter-1)*BS
        feature_vector[six:six+batch_x.shape[0]] = features
        time_store += (time.time()-s)
        # print('time_fw:',time_fw/n_iter, \
        #       'time_store:',time_store/n_iter,\
        #       'time_cpu:',time_cpu/n_iter,
        #       'time_np:',time_np/n_iter)

    if self.dsname == 'CIFAR100_20' or (self.dsname == 'CIFAR100' and args.n_classes == 20):
      labels_vector = list(map(lambda x: _cifar100_to_cifar20_dict[x.item()],labels_vector))

    # feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)

    print("Features shape {}".format(feature_vector.shape),'labels_vector.shape:',labels_vector.shape)
    return feature_vector, labels_vector

  def get_resnet_features(self):
    print('-'*100)
    train_loader,len_tr_loader, test_loader,len_ts_loader = get_data_loaders(self.dsname,cats=self.cats,pickle_path = self.pickle_path,insize=self.insize)
    X_train_feature, y_train = self._inference(train_loader,len_tr_loader)
    X_test_feature, y_test = self._inference(test_loader,len_ts_loader)
    
    return X_train_feature, y_train, X_test_feature, y_test

class LogiticRegressionEvaluator(object):
  def __init__(self, n_features, n_classes,dsname):
    self.log_regression = LogisticRegression(n_features, n_classes).to(device)
    self.scaler = preprocessing.StandardScaler()
    self.dsname = dsname
    self.n_classes = n_classes
  def _normalize_dataset(self, X_train, X_test):
    # print("Standard Scaling Normalizer")
    self.scaler.fit(X_train)
    X_train = self.scaler.transform(X_train)
    X_test = self.scaler.transform(X_test)
    return X_train, X_test

  @staticmethod
  def _sample_weight_decay():
    # We selected the l2 regularization parameter from a range of 45 logarithmically spaced values between 10−6 and 105
    # weight_decay = np.logspace(-6, 5, num=45, base=10.0)
    # weight_decay = np.random.choice(weight_decay)
    weight_decay = 1e-4
    print("Sampled weight decay:", weight_decay)
    return weight_decay

  def eval(self, test_loader):
    correct = 0
    total = 0

    # true = []
    # preds = []
    
    ix = 0
    with torch.no_grad():
      self.log_regression.eval()
      for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits = self.log_regression(batch_x)
        predicted = torch.argmax(logits, dim=1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

        # true.extend(list(batch_y.cpu().numpy()))
        # preds.extend(list(predicted.cpu().numpy()))
        # self.true[ix:ix + len(batch_y)] = batch_y.cpu().numpy()
        # self.preds[ix:ix + len(predicted)] = predicted.cpu().numpy()
        # ix += len(predicted)

      final_acc = 100 * correct / total
      self.log_regression.train()

      nmi,ari,acc = 0,0,0#cluster_metrics(self.true,self.preds,self.n_classes)
      return final_acc,nmi,ari,acc

  def create_data_loaders_from_arrays(self, X_train, y_train, X_test, y_test):
    X_train, X_test = self._normalize_dataset(X_train, X_test)

    train = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).type(torch.long))
    train_loader = torch.utils.data.DataLoader(train, batch_size=396, shuffle=False,num_workers=2)

    test = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).type(torch.long))
    # test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False,num_workers=2)
    test_loader = torch.utils.data.DataLoader(test, batch_size=8000, shuffle=False,num_workers=2)
    print('len(test_loader):',len(test_loader))
    # self.true = np.empty((len(test),))
    # self.preds = np.empty((len(test),))
    # print('true.shape:', self.true.shape)
    return train_loader, test_loader

  def train(self, X_train, y_train, X_test, y_test):
  
    train_loader, test_loader = self.create_data_loaders_from_arrays(X_train, y_train, X_test, y_test)

    weight_decay = self._sample_weight_decay()

    lr = 3e-4
    optimizer = torch.optim.Adam(self.log_regression.parameters(), lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0

    n_epochs = 200

    epoch_accs = []
    stop = 0
    # eval_times = []
    # train_times = []
    for e in tqdm(range(n_epochs)):
    
      # s = time.time()
      for batch_x, batch_y in train_loader:

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()

        logits = self.log_regression(batch_x)

        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()
      # train_times.append(time.time() - s)
      
      # s = time.time()
      epoch_acc,nmi,ari,acc = self.eval(test_loader)
      # eval_times.append(time.time() - s)

      # print(epoch_acc)
      epoch_accs.append(epoch_acc)
      if epoch_acc > best_accuracy:
        stop = 0
        #print("Saving new model with accuracy {}".format(epoch_acc))
        best_accuracy = epoch_acc
        best_nmi,best_ari,best_acc = nmi,ari,acc
        # torch.save(self.log_regression.state_dict(), 'log_regression.pth')
      else:
        stop += 1
      if stop >=10:
        break
    # eval_times = np.array(eval_times)
    # train_times = np.array(train_times)
    # print(len(eval_times),eval_times.mean(),eval_times.std(),len(train_times),train_times.mean(),train_times.std())
    epoch_accs = np.array(epoch_accs)
    print(len(epoch_accs),epoch_accs.mean(),epoch_accs.std(),epoch_accs.min(),epoch_accs.max())
    # print("--------------")
    print("Best accuracy:", best_accuracy,best_nmi,best_ari,best_acc)

def linear_main(lineardataroot,_device,model_dir,model_name,n_classes=None,pickle_path=None,model=None,**kwargs):
  
  global data_root
  data_root = lineardataroot

  global batch_size
  batch_size = kwargs.get('bs',64)

  global device
  device = _device#'cuda' if torch.cuda.is_available() else 'cpu'
  # print("Using device:", device)

  parser = argparse.ArgumentParser()
  # basic experiment setting
  parser.add_argument('model_dir',type=str, default=None)
  parser.add_argument('model_name',type=str, default=None)
  parser.add_argument('--n_classes',type=int, default=None)
  parser.add_argument('--pickle_path',type=str, default=None)
  # args = parser.parse_args(sys.argv[1:])
  args = parser.parse_args([model_dir,model_name])
  args.n_classes = n_classes
  args.pickle_path = pickle_path
  args.dsname = kwargs.get('dsname')
  args.insize = kwargs.get('insize',96)
  # print('args.insize:',args.insize)

  for k,v in sorted(vars(args).items()):
    print('{}: {}'.format(str(k).ljust(20),v ) )
  ###################################################################################################

  config_path = os.path.join(args.model_dir,'config.yaml')
  model_path = os.path.join(args.model_dir,args.model_name)
  # max_iter = int(sys.argv[3])#1200
  out_path = '{}_evals'.format(model_path)

  sys.stdout = open(out_path,'a')
  print('#'*100)
  print('Commande: ',' '.join(sys.argv))

  if args.dsname:
    dsname = args.dsname
  elif args.model_name.startswith('STL10') or args.model_name.startswith('COCO') or args.model_name.startswith('ADE'):
    dsname = 'STL10'
  elif args.model_name.startswith('CIFAR10'):
    dsname = 'CIFAR10'
  elif args.model_name.startswith('CIFAR100_20'):
    dsname = 'CIFAR100_20'
  elif args.model_name.startswith('CIFAR100'):
    dsname = 'CIFAR100'

  global config
  cats=''
  if os.path.exists(config_path):
    config = yaml.load(open(config_path))
    if 'cats' in config['dataset'][config['dataset']['dataset']]:
      cats = config['dataset'][config['dataset']['dataset']]['cats']
    config.update({'conv1_k3':config['conv1_k3'] or (args.model_name.startswith('CIFAR') or args.model_name.startswith('SVHN') )})
  print('model_path:',model_path)
  resnet_feature_extractor = ResNetFeatureExtractor(model_path,dsname,cats=cats,pickle_path=args.pickle_path,model = model,insize=args.insize)

  X_train_feature, y_train, X_test_feature, y_test = resnet_feature_extractor.get_resnet_features()

  if args.n_classes is not None:
    n_classes = args.n_classes
  else:
    n_classes = 100 if dsname == 'CIFAR100' else 20 if dsname == 'CIFAR100_20' else 10
  log_regressor_evaluator = LogiticRegressionEvaluator(n_features=X_train_feature.shape[1], n_classes=n_classes,dsname = dsname)

  log_regressor_evaluator.train(X_train_feature, y_train, X_test_feature, y_test)

  sys.stdout.close()
  sys.stdout = sys.__stdout__

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # basic experiment setting
  parser.add_argument('dataroot',type=str, default=None)
  parser.add_argument('model_dir',type=str, default=None)
  parser.add_argument('model_name',type=str, default=None)
  parser.add_argument('--n_classes',type=int, default=None)
  parser.add_argument('--batch_size',type=int, default=128)
  parser.add_argument('--insize',type=int, default=96)
  parser.add_argument('--pickle_path',type=str, default=None)
  parser.add_argument('--dsname',type=str, default=None)
  args = parser.parse_args(sys.argv[1:])
  # args = parser.parse_args([model_dir,model_name])
  _device = 'cuda' if torch.cuda.is_available() else 'cpu'
  linear_main(args.dataroot, _device,args.model_dir,args.model_name,\
              n_classes = args.n_classes, pickle_path = args.pickle_path, 
              bs = args.batch_size,dsname = args.dsname,
              insize= args.insize)