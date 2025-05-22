import time

import torch
import numpy as np
import h5py
import os
import torch.nn as nn
from torch.autograd.variable import Variable

def eval_loss(model,test_loader,device,args):
    criterion = nn.MSELoss()
    correct = 0
    total_loss = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

            batch_size = batch_x.size(0)
            total += batch_size
            batch_x = Variable(batch_x.float().to(device))
            batch_y = Variable(batch_y.float().to(device))
            batch_x_mark = Variable(batch_x_mark.float().to(device))
            batch_y_mark = Variable(batch_y_mark.float().to(device))
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == 'MS' else 0
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            outputs=outputs[:, -args.pred_len:, :]
            loss = criterion(outputs, batch_y)
            total_loss += loss * batch_size #.item()


    return total_loss / total


def calulate_loss_landscape(model, directions,path,test_loader,device,args):
    setup_surface_file(path,args)
    init_weights = [p.data for p in model.parameters()]

    save_name=path+"/3d_surface_file.h5"
    with h5py.File(save_name, 'r+') as f:
        xcoordinates = f['xcoordinates'][:]
        ycoordinates = f['ycoordinates'][:]
        losses = f["test_loss"][:]
        inds, coords = get_indices(losses, xcoordinates, ycoordinates)
        for count, ind in enumerate(inds):
            print("ind...%s" % ind)
            coord = coords[count]
            overwrite_weights(model, init_weights, directions, coord,device)
            loss= eval_loss(model,test_loader,device,args)
            losses.ravel()[ind] = loss
            f["test_loss"][:] = losses
            f.flush()


def setup_surface_file(path,args):
    xmin, xmax, xnum = -1, 1, 51
    ymin, ymax, ynum = -1, 1, 51
    if args.CL_Strategy:
        surface_path=path+"/3d_surface_file_CL.h5"
    else:
        surface_path = path+"/3d_surface_file.h5"
    # print(surface_path)
    # time.sleep(500)
    if os.path.isfile(surface_path):
        print("%s is already set up" % "3d_surface_file.h5")

        return

    with h5py.File(surface_path, 'a') as f:
        print("create new 3d_sureface_file.h5")

        xcoordinates = np.linspace(xmin, xmax, xnum)
        f['xcoordinates'] = xcoordinates

        ycoordinates = np.linspace(ymin, ymax, ynum)
        f['ycoordinates'] = ycoordinates

        shape = (len(xcoordinates), len(ycoordinates))
        losses = -np.ones(shape=shape)


        f["test_loss"] = losses


        return


def get_indices(vals, xcoordinates, ycoordinates):
    inds = np.array(range(vals.size))
    inds = inds[vals.ravel() <= 0]

    xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
    s1 = xcoord_mesh.ravel()[inds]
    s2 = ycoord_mesh.ravel()[inds]

    return inds, np.c_[s1, s2]


def overwrite_weights(model, init_weights, directions, step,device):
    dx = directions[0]
    dy = directions[1]
    changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]

    for (p, w, d) in zip(model.parameters(), init_weights, changes):
        p.data = w.to(device) + torch.Tensor(d).to(device)
