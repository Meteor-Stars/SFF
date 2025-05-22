import numpy as np
import torch

# class Metrics_Calculation():
#     def __init__(self, args=None):
#         self.args=args
#         self.li_loss=torch.nn.L1Loss()
#     def cal_metrics_val(self,pred, true):
#         fund_metric = {}
#         mae = MAE(pred, true)
#         mse = MSE(pred, true)
#         rmse = RMSE(pred, true)
#         corr = CORR(pred, true)
#         rse = RSE(pred, true)
#         # pmae = PMAE(pred, true)
#         wmape=calculate_wmape(pred,true)
#         fund_metric['mae']=mae
#         fund_metric['mse']=mse
#         fund_metric['rmse']=rmse
#         fund_metric['corr']=np.mean(corr)
#         fund_metric['rse']=rse
#         # fund_metric['pmae'] = pmae
#         fund_metric['wmape']=wmape
#
#         for k, v in fund_metric.items():
#             # print(type(v))
#             # fund_metric[k] = float(np.round(np.float32(v),5))
#
#             fund_metric[k] =float(v)
#
#         return fund_metric
#
#
# def calculate_wmape(pred, true, weights=None):
#
#     if weights is None:
#         weights = np.ones_like(pred)
#
#     numerator = np.sum(np.abs(pred - true) * weights)
#     denominator = np.sum(np.abs(true) * weights)
#
#     # wmape = numerator / denominator * 100
#     wmape = numerator / denominator
#     return wmape


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def NRMSE(pred, true):
    return np.sqrt(MSE(pred, true)) / np.mean(np.abs(true))


def WAPE(pred, true):
    return np.mean(np.abs(pred - true)) / np.mean(np.abs(true))



def metric(pred, true):
    metric_all={}

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    metric_all['mae'] = mae
    metric_all['mse'] = mse
    metric_all['rmse'] = rmse
    metric_all['mape'] = mape
    metric_all['mspe'] = mspe


    for k, v in metric_all.items():
        metric_all[k] = float(v)

    return metric_all

