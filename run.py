import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist

from exp.exp_forecast import Exp_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_imputation import Exp_Imputation
from utils.tools import HiddenPrints

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Large Time Series Model')

    # basic config
    # parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
    #                     help='task name, options:[forecast, imputation, anomaly_detection]')
    # parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    # parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    # parser.add_argument('--model', type=str, required=True, default='Timer',
    #                     help='model name, options: [Timer TrmEncoder]')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # data loader
    # parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # model define
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--ckpt_path', type=str, default='', help='ckpt file')
    parser.add_argument('--finetune_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--finetune_rate', type=float, default=0.1, help='finetune ratio')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')

    parser.add_argument('--patch_len', type=int, default=24, help='input sequence length')
    parser.add_argument('--subset_rand_ratio', type=float, default=1, help='mask ratio')
    parser.add_argument('--data_type', type=str, default='custom', help='data_type')

    parser.add_argument('--decay_fac', type=float, default=0.75)

    # cosin decay
    parser.add_argument('--cos_warm_up_steps', type=int, default=100)
    parser.add_argument('--cos_max_decay_steps', type=int, default=60000)
    parser.add_argument('--cos_max_decay_epoch', type=int, default=10)
    parser.add_argument('--cos_max', type=float, default=1e-4)
    parser.add_argument('--cos_min', type=float, default=2e-6)

    # weight decay
    parser.add_argument('--use_weight_decay', type=int, default=0, help='use_post_data')
    parser.add_argument('--weight_decay', type=float, default=0.01)

    # autoregressive configs
    parser.add_argument('--use_ims', action='store_true', help='Iterated multi-step', default=False)
    parser.add_argument('--output_len', type=int, default=96, help='output len')
    parser.add_argument('--output_len_list', type=int, nargs="+", help="output_len_list")

    # train_test
    parser.add_argument('--train_test', type=int, default=1, help='train_test')
    parser.add_argument('--is_finetuning', type=int, default=1, help='status')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    args = parser.parse_args()


    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_multi_gpu:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64209")
        hosts = int(os.environ.get("WORLD_SIZE", "8"))  # number of nodes
        rank = int(os.environ.get("RANK", "0"))  # node id
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()  # gpus per node
        args.local_rank = local_rank
        print(
            'ip: {}, port: {}, hosts: {}, rank: {}, local_rank: {}, gpus: {}'.format(ip, port, hosts, rank, local_rank,
                                                                                     gpus))
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts, rank=rank)
        print('init_process_group finished')
        torch.cuda.set_device(local_rank)
    args.root_path='./datasets/'
    data_name='ETTh1'
    if data_name=='ETTh1':
        args.data_path='ETT-small/ETTh1.csv'
    elif data_name=='ETTh2':
        args.data_path='ETT-small/ETTh2.csv'
    elif data_name=='ETTm1':
        args.data_path='ETT-small/ETTm1.csv'
    elif data_name=='ETTm2':
        args.data_path='ETT-small/ETTm2.csv'
    elif data_name=='weather':
        args.data_path='weather/weather.csv'
    elif data_name == 'electricity':
        args.data_path = 'electricity/electricity.csv'
    elif data_name=='traffic':
        args.data_path='traffic/traffic.csv'
    elif data_name=='exchange_rate':
        args.data_path='exchange_rate.csv'

    args.is_training=1
    args.ckpt_path='/data_new/daroms/paroms/LTSM/checkpoints/Timer_forecast_1.0.ckpt'
    args.model='Timer'
    dataset_name=args.data_path.split('/')[-1].split('.')[0]
    args.data=dataset_name
    args.model_id=dataset_name
    args.gpu=4
    extra_ = ''
    args.record=True
    args.task_name = 'forecast'
    args.checkpoints = './checkpoints/' + dataset_name + '/' + args.model + '/' + 'random_seed_' + str(args.seed)
    args.training_from_scratch=False
    args.smoothed_full_finetuning=False
    args.LP=False #linear probing
    args.LPFF=False #linear probing first then full fine-tuning
    args.ZeroShot=False
    if args.LP:
        args.smoothed_full_finetuning=False
        extra_ += '_LP'
    if args.LPFF:
        args.smoothed_full_finetuning = False
        extra_ += '_LPFF'

    if args.smoothed_full_finetuning:
        args.alpha = 0.5
        args.training_from_scratch=False
        extra_+='_SFF'
    if not args.training_from_scratch and not args.smoothed_full_finetuning:
        args.standard_finetuning=True
        extra_ += '_StandFF'
    if args.training_from_scratch:
        extra_+='_TFS'
    if args.ZeroShot:
        extra_+='_ZeroShot'
    if args.task_name=='forecast':
        if 'ETTh' in data_name or 'exchange' in data_name:
            args.batch_size = 2048
        elif 'traffic' in data_name:
            args.batch_size = 2048 * 3
        else:
            args.batch_size = 2048 * 3
        args.seq_len = 672
        args.label_len = 576
        args.pred_len = 96
        args.output_len = 96
        args.patch_len = 96
        args.subset_rand_ratio = 1
        args.e_layers=8
        args.factor=3
        args.d_model=1024
        args.d_ff=2048
        args.batch_size=2048
        args.learning_rate=3e-5
        args.num_workers=4
        args.train_test=1
        Exp = Exp_Forecast
    elif args.task_name=='imputation':
        args.d_model = 256
        args.d_ff = 512
        args.e_layers = 4
        args.patch_len = 24
        args.seq_len= 192
        args.label_len= 0
        args.pred_len=192
        args.factor=3
        args.batch_size=16
        if data_name=='traffic':
            args.batch_size=32
        else:
            args.batch_size=128
        args.train_test=1
        args.train_epochs=10
        args.mask_rate=0.25
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        args.seq_len = 768
        args.d_model = 256
        args.d_ff = 512
        args.e_layers = 4
        args.patch_len = 96
        args.pred_len=0
        args.batch_size=128
        args.subset_rand_ratio=1.0
        args.train_epochs=10
        args.train_test=1
        Exp = Exp_Anomaly_Detection
    else:
        raise ValueError('task name not found')

    with HiddenPrints(int(os.environ.get("LOCAL_RANK", "0"))):
        print('Args in experiment:')
        print(args)
        if args.is_finetuning:
            for ii in range(args.itr):
                # setting record of experiments
                setting = '{}_{}_seq{}_pl{}'.format(
                    args.model,
                    args.data,
                    args.seq_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des,
                    ii)
                # setting += datetime.now().strftime("%y-%m-%d_%H-%M-%S")
                setting+=extra_

                exp = Exp(args)  # set experiments
                if not args.ZeroShot:
                    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                    if args.LP:
                        exp.finetune_LP(setting)
                    elif args.LPFF:
                        exp.finetune_LP(setting)
                        exp.finetune_FF(setting)
                    else:
                        exp.finetune(setting)

                    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.test(setting)
                    torch.cuda.empty_cache()
                else:
                    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.test(setting)
                    torch.cuda.empty_cache()
        else:
            ii = 0
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                ii)

            setting += datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            exp = Exp(args)  # set experiments
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
            torch.cuda.empty_cache()
