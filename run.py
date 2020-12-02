

import sys,os,time,argparse
import yaml
from train import MIXEM

import socket

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path',type=str, default=None)
    parser.add_argument('--log_dir',type=str, default=None)
    parser.add_argument('--gpus',type=str, default=None)
    parser.add_argument('--start_epoch',type=int, default=1)
    parser.add_argument('--master_batch_size',type=int, default=-1)
    parser.add_argument('--init_from',type=str, default=None)
    parser.add_argument('--dataroot',type=str, default=None)
    parser.add_argument('--lineardataroot',type=str, default=None)
    parser.add_argument('--dsname',type=str, default=None)
    parser.add_argument('--epochs',type=int, default=None)
    parser.add_argument('--num_workers',type=int, default=4)
    parser.add_argument('--fixed_lr',action='store_true')
    parser.add_argument('--no_eval',action='store_true')
    parser.add_argument('--resume',action='store_true')
    
    args = parser.parse_args(sys.argv[1:])
    
    for k,v in sorted(vars(args).items()):
        print('{}: {}'.format(str(k).ljust(20),v ) )
    ###################################################################################################

    if args.resume:
        args.config_path = os.path.join(args.log_dir,'checkpoints','config.yaml')
        with open(os.path.join(args.log_dir, 'checkpoints','checkpoint'),'r') as _if:
            args.init_from = _if.readline().strip('\n')
        print('init_from:',args.init_from)
            

    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)

    args.gpus = list(map(int, args.gpus.split(',')))

    _lr = config['learning_rate']
    config.update({'log_dir': args.log_dir, 
                    'config_path':args.config_path,
                    'gpus':args.gpus,   
                    'master_batch_size':args.master_batch_size,
                    'dataroot':args.dataroot,
                    'resume':args.resume,
                    'start_epoch':  int(args.init_from.split('.')[0].split('_')[-1])+1 if args.resume else args.start_epoch
                    })
    config.update({'lineardataroot': args.lineardataroot or config.get('lineardataroot',None)})
    config.update({'init_from': args.init_from or config['init_from']})
    config.update({'epochs':args.epochs or config['epochs']})
    config.update({'no_eval':args.no_eval or config.get('no_eval',False)})
    config.update({'fixed_lr':args.fixed_lr or config.get('fixed_lr',False)})

    config['dataset'].update({'dataset': args.dsname or config['dataset']['dataset']})
    dsname = config['dataset']['dataset']
    print('dsname:',config['dataset']['dataset'])
    config['dataset'][dsname].update({'num_workers': args.num_workers or config['dataset'][dsname]['num_workers']})

    config.update({'conv1_k3':config['conv1_k3'] or (dsname.startswith('CIFAR') )})
    if args.resume:
        config.update({'init_from': os.path.join(config['log_dir'],'checkpoints',config['init_from']) })
        assert os.path.exists(config['init_from'])
    
    if config['init_from'] == 'None':
        config.update({'init_from':None})
    
    model_config = '_{}_{}'.format(config['model']['base_model'],config['model']['out_dim'])
    con1k3 = '_conv1k3' if config['conv1_k3'] else ''
    model_config += con1k3
    
    temp = '_temp_{}'.format(config['loss']['temperature'])
    
    optconfig = '_LR_{}'.format(config['learning_rate'])
    if config['fixed_lr']:
        optconfig += '_FIXED'
    if config['weight_decay'] != 1e-5:
        optconfig += '_WDECAY_{}'.format(config['weight_decay']) 
    
    finetune = ''
    if not args.resume:
        finetune = '_init_form_{}_{}'.format(config['init_from'].split('/')[-3].split('_')[-1],config['init_from'].split('/')[-1] ) if config['init_from'] is not None else ''
        if args.start_epoch > 1:
            finetune += '_STARTEP_{}'.format(args.start_epoch)
    mixture = ''
    if 'Mixture' in config['model']:
        config['model']['Mixture'].update({'temperature': config['model']['Mixture'].get('temperature',1.),\
                                        'pimax_loss_w':config['model']['Mixture'].get('pimax_loss_w',0),\
                                        'w_push':config['model']['Mixture'].get('w_push',0),\
                                        'w_pull':config['model']['Mixture'].get('w_pull',0)
                                        })
        pimax_loss = ''
        if config['model']['Mixture']['pimax_loss_w']>0:
            pimax_loss = '_PIMAXL_{}'.format(config['model']['Mixture']['pimax_loss_w'])
        pushw = config['model']['Mixture']['w_push']
        pullw = config['model']['Mixture']['w_pull']
        pp = 'PP_{}_{}'.format(pushw,pullw) if pushw > 0 or pullw > 0 else ''
            
            

        mixture = '_MIXTURE_{}_{}_ENT_{}{}'.format(config['model']['Mixture']['n_comps'],\
                                            pp,config['model']['Mixture']['w_entropy'],pimax_loss)
                    


    log_dir_label = '{}{}{}_{}{}{}{}'.format(dsname,mixture,\
                                                model_config,\
                                                config['batch_size'],\
                                                temp,optconfig,finetune)
    if args.resume:
        log_dir = config['log_dir']
        if log_dir.endswith('/'):
            log_dir = log_dir[:-1]
        print('log_dir:'.ljust(30),log_dir.split('/')[-1])
        print('log_dir_label:'.ljust(30),log_dir_label)
        assert log_dir.split('/')[-1].startswith(log_dir_label)
    else:
        log_dir = '{}/{}_gpus_{}_{}'.format(config['log_dir'],log_dir_label,','.join(map(str,args.gpus)),int(time.time()))
        config.update({'log_dir':log_dir})
    
    print('log_dir:',config['log_dir'])
    mixem = MIXEM(config)
    mixem.train()

if __name__ == "__main__":
    main()
    print('End')
