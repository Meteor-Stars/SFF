import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual
import json

warnings.filterwarnings('ignore')


class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)

    def _build_model(self):
        # model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = self.model_dict[self.args.model].Model(self.args).float()
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        else:
            self.args.device = self.device
            if self.args.re_train:
                self.args.ckpt_path = 'random'
                model = self.model_dict[self.args.model].Model(self.args)
            else:
                model = self.model_dict[self.args.model].Model(self.args)
            if self.args.smoothed_full_finetuning:

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
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        preds = []
        trues = []
        masks = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                """
                B = batch size
                T = seq len
                N = number of features
                """
                assert T % self.args.patch_len == 0
                mask = torch.rand((B, T // self.args.patch_len, N)).to(self.device)
                mask = mask.unsqueeze(2).repeat(1, 1, self.args.patch_len, 1)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                mask = mask.view(mask.size(0), -1, mask.size(-1))
                mask[:, :self.args.patch_len, :] = 1  # first patch is always observed
                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, None, None, mask)

                f_dim = -1 if self.args.features == 'MS' else 0

                if self.args.use_ims:
                    outputs = outputs[:, :-self.args.patch_len, f_dim:]
                else:
                    outputs = outputs[:, self.args.patch_len:, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_x[:, self.args.patch_len:, f_dim:].detach().cpu()
                mask = mask[:, self.args.patch_len:, f_dim:].detach().cpu()


                preds.append(pred)
                trues.append(true)
                masks.append(mask.detach().cpu())

                loss = criterion(pred[mask == 0], true[mask == 0])
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        print('valid',preds.shape)
        mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
        # print('mse:{}, mae:{}'.format(mse, mae))
        total_loss_dict={}
        total_loss_dict['mae']=float(mae)
        total_loss_dict['mse']=float(mse)
        total_loss_dict['rmse']=float(rmse)
        total_loss_dict['mape']=float(mape)
        total_loss_dict['mspe']=float(mspe)
        print(total_loss_dict)
        return total_loss_dict

    def finetune(self, setting):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        # print('okok')
        # time.sleep(500)
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        train_len = train_data.__len__() // self.args.batch_size * self.args.subset_rand_ratio
        train_loss_all_dict={}
        valid_loss_all_dict={}
        test_loss_all_dict={}
        min_best_loss=100
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                if i > train_len:
                    break
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                assert T % self.args.patch_len == 0
                mask = torch.rand((B, T // self.args.patch_len, N)).to(self.device)
                mask = mask.unsqueeze(2).repeat(1, 1, self.args.patch_len, 1)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                mask = mask.view(mask.size(0), -1, mask.size(-1))
                mask[:, :self.args.patch_len, :] = 1  # first patch is always observed
                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, None, None, mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                if self.args.use_ims:
                    outputs = outputs[:, :-self.args.patch_len, f_dim:]
                else:
                    outputs = outputs[:, self.args.patch_len:, f_dim:]

                true = batch_x[:, self.args.patch_len:, f_dim:]
                loss = criterion(outputs[mask[:, self.args.patch_len:, f_dim:] == 0],
                                 true[mask[:, self.args.patch_len:, f_dim:] == 0])
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            if vali_loss['mse'] < min_best_loss:
                min_best_loss = vali_loss['mse']
                torch.save(self.model.backbone.state_dict(), path + '/' + 'checkpoint' + '.pth')
            test_loss=0
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss['mse'], test_loss['mse']))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss['mse']))
            train_loss_all_dict[epoch] = train_loss
            valid_loss_all_dict[epoch]=vali_loss
            test_loss_all_dict[epoch] = test_loss

            json_record_loss_train = json.dumps(train_loss_all_dict, indent=4)
            json_record_loss_test = json.dumps(test_loss_all_dict, indent=4)
            if self.args.record:
                with open(path + '/record_all_loss_train' + '.json', 'w') as json_file:
                    json_file.write(json_record_loss_train)
                with open(path + '/record_all_loss_test' + '.json', 'w') as json_file:
                    json_file.write(json_record_loss_test)

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        masks = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                assert T % self.args.patch_len == 0
                mask = torch.rand((B, T // self.args.patch_len, N)).to(self.device)
                mask = mask.unsqueeze(2).repeat(1, 1, self.args.patch_len, 1)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                mask = mask.view(mask.size(0), -1, mask.size(-1))
                mask[:, :self.args.patch_len, :] = 1  # first patch is always observed
                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, None, None, mask)

                # eval
                f_dim = -1 if self.args.features == 'MS' else 0
                if self.args.use_ims:
                    outputs = outputs[:, :-self.args.patch_len, f_dim:]
                else:
                    outputs = outputs[:, self.args.patch_len:, f_dim:]

                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true = batch_x[:, self.args.patch_len:, f_dim:].detach().cpu().numpy()
                mask = mask[:, self.args.patch_len:, f_dim:]
                preds.append(pred)
                trues.append(true)
                masks.append(mask.detach().cpu())

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        print('test shape:', preds.shape, trues.shape)
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
        print('mse:{}, mae:{}'.format(mse, mae))

        return
