import sys, os, time, shutil, yaml
import torch
import numpy as np
np.random.seed(0)
torch.manual_seed(0)
from resnet import ResNet
from data_aug.dataset_wrapper import DataSetWrapper
from torch.utils.tensorboard import SummaryWriter
from nt_xent import NTXentLoss
import torch.nn.functional as F
from collections import Counter,OrderedDict
from linear import linear_main
from data_parallel import DataParallel

apex_support = False

def _save_config_file(model_checkpoints_folder,config):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yaml'), 'w') as of:
        yaml.dump(config, of, default_flow_style=False)

def _get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running on:", device)
    return device

def PairEnum(x,mask=None):
    '''Source: https://github.com/k-han/AutoNovel/blob/master/utils/util.py'''
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0),1)
    x2 = x.repeat(1,x.size(0)).view(-1,x.size(1))
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2

class MIXEM(object):

    def __init__(self, config):
        self.config = config
        self.n_gpu = len(self.config['gpus'])
        self.temperature = self.config['loss']['temperature']
        self.device = _get_device()
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        self.batch_size = config['batch_size']
        self.lineardataroot = config['lineardataroot']
        self.dsname = config['dataset']['dataset']
        
        self.n_comps = config['model']['Mixture']['n_comps']
        self.pushw = self.config['model']['Mixture'].get('w_push',0)
        self.pullw = self.config['model']['Mixture'].get('w_pull',0)
        if self.pushw>0 or self.pullw>0:
            self.zeroz_x = torch.zeros(( self.n_comps, config['model']['out_dim']) ).to(self.device)
            self.zeroz_y = torch.arange( self.n_comps ).to(self.device)

        self.nt_xent_criterion = NTXentLoss(self.config,self.device, self.batch_size, **config['loss'])

        self.dataset = DataSetWrapper(self.dsname,self.batch_size,**config['dataset'][self.dsname],
                    dataroot= config['dataroot'])


    def mixem_loss(self, model, xs, epoch_counter,n_iter,stage):

        xis = xs[0]; xjs = xs[1]

        ris, (pi_i_orig,mixture_i) = model(xis)
        rjs, (pi_j_orig,mixture_j) = model(xjs)

        mixture_i = F.normalize(mixture_i, dim=2)
        mixture_j = F.normalize(mixture_j, dim=2)
        pi_i = F.softmax(pi_i_orig,dim=1).unsqueeze(2)
        pi_j = F.softmax(pi_j_orig,dim=1).unsqueeze(2)

        __probs_i,__comps_i = torch.max(pi_i.detach().squeeze(),dim=1)
        __probs_j,__comps_j = torch.max(pi_j.detach().squeeze(),dim=1)
        tmp = torch.logical_or(__probs_i>.7,__probs_j>.7)

        zis = (pi_i*mixture_i).sum(dim=1)
        zjs = (pi_j*mixture_j).sum(dim=1)
        loss = 0 
        loss = self.nt_xent_criterion(zis, zjs,n_iter,self.writer,stage,epoch_counter)

        self.writer.add_histogram('Max_comp_pi_i_{}'.format(stage),torch.max(pi_i.squeeze(),dim=1)[1],global_step=n_iter)
        self.writer.add_histogram('Max_comp_pi_i_prob_{}'.format(stage),torch.max(pi_i.squeeze(),dim=1)[0],global_step=n_iter)

        w_entropy =  self.config['model']['Mixture']['w_entropy']
        if w_entropy>0:

            pi_i_ent = F.softmax(pi_i_orig, dim=1).mean(dim=0)
            ent1 = ( pi_i_ent*torch.log(pi_i_ent)).sum()*w_entropy
            pi_j_ent = F.softmax(pi_j_orig, dim=1).mean(dim=0)
            ent2 = ( pi_j_ent*torch.log(pi_j_ent)).sum()*w_entropy

            ent = ent1+ent2
            self.writer.add_scalar('Entropy/{}'.format(stage), ent,global_step=n_iter)
            loss += ent
        
        pimax_loss_w = self.config['model']['Mixture'].setdefault('pimax_loss_w',0)
        if pimax_loss_w >0:
            pimax_l = 0
            pi_i_max = torch.max(pi_i.squeeze(),dim=1)[0]
            pimax_l+= (1.-pi_i_max).mean()*pimax_loss_w
            pi_j_max = torch.max(pi_j.squeeze(),dim=1)[0]
            pimax_l+= (1.-pi_j_max).mean()*pimax_loss_w
            self.writer.add_scalar('PiMaxLoss/{}'.format(stage), pimax_l,global_step=n_iter)
            loss += pimax_l

    
        pushw = self.pushw; pullw = self.pullw
        if pushw>0 or pullw>0:
            pi_i = F.softmax(pi_i_orig, dim=1)
            pi_j = F.softmax(pi_j_orig, dim=1)
            pi_i_max = torch.max(pi_i.squeeze().detach(),dim=1)[1]
            pi_j_max = torch.max(pi_j.squeeze().detach(),dim=1)[1]
            pull_loss = 0 
            push_loss = 0 
            for x,y in [(zis,pi_i_max),(zjs,pi_j_max)]:
                x = torch.cat((x,self.zeroz_x))
                y = torch.cat((y,self.zeroz_y))
                means = torch.stack([x[y==c].mean(dim=0)  for c in range(self.n_comps)],dim=0)
                pull = torch.stack([ torch.mean(torch.linalg.norm(x[y==c]-means[c],ord=2,dim=1)) for c in range(self.n_comps)],dim=0)
                pull_loss += pull.sum()/self.n_comps*pullw
                m1,m2 = PairEnum(means)
                push = torch.linalg.norm(m1-m2,ord=2,dim=1)
                push = -1*push.sum()/(self.n_comps**2-self.n_comps)
                push_loss += push*pushw

            loss += pull_loss
            loss += push_loss
            self.writer.add_scalar('Pull_Loss/{}'.format(stage), pull_loss,global_step=n_iter)
            self.writer.add_scalar('Push Loss/{}'.format(stage), push_loss,global_step=n_iter)

        return loss


    def train(self):

        timers = Counter()
        train_loader, valid_loader = self.dataset.get_data_loaders()
        model = ResNet(**self.config["model"],conv1_k3 = self.config['conv1_k3'])
        model = self._load_pre_trained_weights(model)
        init_lr = eval(self.config['learning_rate'])
        print('init_lr:',init_lr)
        optimizer = torch.optim.Adam(model.parameters(), init_lr, weight_decay=eval(self.config['weight_decay']))
        min_lr=0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'], eta_min=min_lr,last_epoch=-1)
        start_epoch = self.config['start_epoch']
        print('start_epoch:',start_epoch)
        if not self.config.get('fixed_lr',False):
            for _ in range(self.config.get('warmup',10),start_epoch,1):
                scheduler.step()
                                                                            
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        _save_config_file(model_checkpoints_folder,self.config)

        print("iters per epoch:", len(train_loader))
        if self.config['init_from'] is None:
            model_file_name = '{}_{}.pth'.format(self.config['dataset']['dataset'],0)
            save_model(model,model_checkpoints_folder,model_file_name)


        if self.n_gpu>1:
            print('parallelizing the model ...')
            bs = self.batch_size
            master_bs = self.config['master_batch_size']
            if master_bs == -1:
                master_bs = bs // self.n_gpu
            rest_bs = (bs - master_bs)
            chunk_sizes = [master_bs]
            for i in range(self.n_gpu - 1):
                slave_chunk_size = rest_bs // (self.n_gpu - 1)
                if i < rest_bs % (self.n_gpu - 1):
                    slave_chunk_size += 1
                chunk_sizes.append(slave_chunk_size)
            model = DataParallel(model,device_ids = [i for i in range(self.n_gpu)],chunk_sizes=chunk_sizes)
        model = model.to(self.device)

        n_iter = len(train_loader)*(start_epoch-1)
        name_tmp = 2
        if self.config['init_from'] is not None and '_' in self.dsname:
            name_tmp += 1
        if self.config['init_from'] is not None and len(self.config['init_from'].split('/')[-1].split('.')[0].split('_')) > name_tmp:
            n_iter = int(self.config['init_from'].split('/')[-1].split('.')[0].split('_')[-2])
        valid_n_iter = len(valid_loader)*(start_epoch-1)
        best_valid_loss = np.inf

        self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
        for epoch_counter in range(start_epoch,self.config['epochs']+1):

            epoch_stime = int(time.time()*1000)
            model.train()

            print('Epochs:',epoch_counter, 'LR:',scheduler.get_last_lr())
            for xs,_ in train_loader:
                iter_s = time.time()*1000
                optimizer.zero_grad()

                s = int(time.time()*1000)
                xs = [_xsi.to(self.device) for _xsi in xs]
                t = int(time.time()*1000) -s
                timers['to_device'] += t

                s = int(time.time()*1000)
                loss = self.mixem_loss(model, xs ,epoch_counter, n_iter,'Train')
                t = int(time.time()*1000) -s
                timers['loss'] += t

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                s = int(time.time()*1000)
                loss.backward()
                t = int(time.time()*1000) -s
                timers['backward'] += t

                n_iter %10 == 0 and print('n_iter:', n_iter , 'loss:' , loss.item())
                
                optimizer.step()

                n_iter += 1
                t = int(time.time()*1000 -iter_s)
                timers['iter'] += t
                

                if n_iter %10 == 0:
                    self.print_timers(timers,epoch_counter-1,n_iter)

            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss =0
                valid_loss = self._validate(model if self.n_gpu<=1 else model.module, valid_loader,epoch_counter)
                if epoch_counter %1 == 0 and not self.config.get('no_eval',False):
                    best_valid_loss = (valid_loss < best_valid_loss and valid_loss) or best_valid_loss
                    model_file_name = '{}_{}.pth'.format(self.config['dataset']['dataset'],epoch_counter)
                    save_model(model.module if self.n_gpu > 1 else model,model_checkpoints_folder,model_file_name)
                    
                    linear_args = [self.lineardataroot, self.device,model_checkpoints_folder,model_file_name]
                    linear_kwargs = {'model':model.module if self.n_gpu > 1 else model}
                    linear_main(*linear_args, **linear_kwargs)

                    for ii in range(1,epoch_counter,1):
                        to_del_path = os.path.join(model_checkpoints_folder, '{}_{}.pth'.format(self.config['dataset']['dataset'],ii))
                        if ii %10 != 0 and os.path.exists(to_del_path):
                            os.remove(to_del_path)

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            if epoch_counter >= 10 and not self.config.get('fixed_lr',False):
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)

            epoch_time = int(time.time()*1000) - epoch_stime
            timers['epoch'] += epoch_time
            self.print_timers(timers,epoch_counter,n_iter)
            
    def print_timers(self,timers,epoch_counter,n_iter):
        times_msg  = 'Timers (ms): Epoch {} Iter {}'.format(epoch_counter,n_iter)
        for _timer,times in timers.items():
            if _timer == 'epoch':
                if epoch_counter >= 1:
                    times_msg += '\t{}: {}'.format(_timer, int(times/epoch_counter))
            else:
                times_msg += '\t{}: {}'.format(_timer, int(times/n_iter))
        print(times_msg)
        print('-'*100)

    def _load_pre_trained_weights(self, model):
        try:
            state_dict = torch.load(self.config['init_from'])
            state_dict_c = OrderedDict()
            for k in state_dict:
                kk = k
                if kk.startswith('module.'):
                    kk = kk.replace('module.','')
                state_dict_c.update({kk:state_dict[k]})
            model.load_state_dict(state_dict_c,strict=False)
            print("Loaded pre-trained model with success.")
        except:
            print("Pre-trained weights not found. Training from scratch.")
        return model

    def _validate(self, model, valid_loader,epoch_counter):

        # validation steps
        with torch.no_grad():
            
            model.eval()

            valid_loss = 0.0
            for counter, (xs, _) in enumerate(valid_loader):

                xs = [_xsi.to(self.device) for _xsi in xs]
                loss = self.mixem_loss(model, xs, epoch_counter,epoch_counter*len(valid_loader) + counter,'Val')
                valid_loss += loss.item()
            try:
                valid_loss /= (counter+1)
            except UnboundLocalError:
                pass
        return valid_loss

def save_model(model,model_checkpoints_folder,model_file_name):
    model_path = os.path.join(model_checkpoints_folder, model_file_name)
    torch.save(model.state_dict(), model_path)
    with open(os.path.join(model_checkpoints_folder,'checkpoint'),'w') as of:
        of.write(model_file_name)
