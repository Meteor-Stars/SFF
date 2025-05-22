import os
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, visual, LargeScheduler, attn_map,adjust_learning_rate,adjust_learning_ratev2

from utils.metrics import MAPE_Fund
import json
warnings.filterwarnings('ignore')
import copy
from .directions import create_random_directions
from .calc_loss import calulate_loss_landscape

class Exp_Forecast(Exp_Basic):

    def _build_model(self):
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = self.model_dict[self.args.model].Model(self.args)
            model = DDP(model.cuda(), device_ids=[self.args.local_rank], find_unused_parameters=True)
        else:
            self.args.device = self.device
            if self.args.re_train:
                self.args.ckpt_path = 'random'
                model = self.model_dict[self.args.model].Model(self.args)
            else:
                model = self.model_dict[self.args.model].Model(self.args)
            if self.args.random_w:
                self.args.ckpt_path = 'random'
                model2=self.model_dict[self.args.model].Model(self.args)
                alpha=self.args.alpha
                with torch.no_grad():
                    for param1, param2 in zip(model.parameters(), model2.parameters()):
                        param1.copy_((param1*alpha + param2*(1-alpha)))

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.use_weight_decay:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            # model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def test(self, setting, test=0):
        test_data, test_loader = data_provider(self.args, flag='test')
        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        attns = []
        folder_path = './test_results/' + setting + '/' + self.args.data_path + '/' + f'{self.args.output_len}/'
        if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(folder_path)
        self.model.eval()
        if self.args.output_len_list is None:
            self.args.output_len_list = [self.args.output_len]
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(path)
        best_model_path=path + '/' + 'checkpoint_whole' + '.pth'
        model_init=copy.deepcopy(self.model)

        preds_list = [[] for _ in range(len(self.args.output_len_list))]
        trues_list = [[] for _ in range(len(self.args.output_len_list))]
        self.args.output_len_list.sort()

        model_final=copy.deepcopy(self.model)

        if self.args.plot_landscape:
            rand_directions = create_random_directions(model_init,self.device)
            calulate_loss_landscape(model_final, rand_directions, path, test_loader, self.device, self.args)
            return


        return
