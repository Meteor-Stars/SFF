import time

import torch


def create_random_directions(model,device):
    x_direction = create_random_direction(model,device)
    y_direction = create_random_direction(model,device)

    return [x_direction, y_direction]


def create_random_direction(model,device):
    weights = get_weights(model)
    direction = get_random_weights(weights,device)
    normalize_directions_for_weights(direction, weights)

    return direction


def get_weights(model):

    return [p.data for p in model.parameters()]

import math
def get_random_weights(weights,device):
    return [torch.randn(w.size()).to(device)* math.sqrt(8)+1 for w in weights]

def normalize_direction(direction, weights):
    for d, w in zip(direction, weights):
        d.mul_(w.norm() / (d.norm() + 1e-10))


def normalize_directions_for_weights(direction, weights):
    assert (len(direction) == len(weights))
    for d, w in zip(direction, weights):
        normalize_direction(d, w)
