import torch
import torch.nn.functional as F
from torch import nn
import logging
import argparse
import numpy as np
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from itertools import product

from tqdm import trange
import functions_hv_grad_3d
from functions_evaluation import compute_hv_in_higher_dimensions as compute_hv
from functions_evaluation import fastNonDominatedSort
from functions_hv_grad_3d import grad_multi_sweep_with_duplicate_handling

class HvMaximization(object):
    """
    Mo optimizer for calculating dynamic weights using higamo style hv maximization
    based on Hao Wang et al.'s HIGA-MO
    uses non-dominated sorting to create multiple fronts, and maximize hypervolume of each
    """

    def __init__(self, n_mo_sol, n_mo_obj, ref_point, obj_space_normalize=True):
        super(HvMaximization, self).__init__()
        self.name = 'hv_maximization'
        self.ref_point = np.array(ref_point)
        self.n_mo_sol = n_mo_sol
        self.n_mo_obj = n_mo_obj
        self.obj_space_normalize = obj_space_normalize

    def compute_weights(self, mo_obj_val):
        n_mo_obj = self.n_mo_obj
        n_mo_sol = self.n_mo_sol

        # non-dom sorting to create multiple fronts
        hv_subfront_indices = fastNonDominatedSort(mo_obj_val)
        dyn_ref_point = 1.1 * np.max(mo_obj_val, axis=1)
        for i_obj in range(0, n_mo_obj):
            dyn_ref_point[i_obj] = np.maximum(self.ref_point[i_obj], dyn_ref_point[i_obj])
        number_of_fronts = np.max(hv_subfront_indices) + 1  # +1 because of 0 indexing

        obj_space_multifront_hv_gradient = np.zeros((n_mo_obj, n_mo_sol))
        for i_fronts in range(0, number_of_fronts):
            # compute HV gradients for current front
            temp_grad_array = grad_multi_sweep_with_duplicate_handling(mo_obj_val[:, (hv_subfront_indices == i_fronts)],
                                                                       dyn_ref_point)
            obj_space_multifront_hv_gradient[:, (hv_subfront_indices == i_fronts)] = temp_grad_array

        # normalize the hv_gradient in obj space (||dHV/dY|| == 1)
        normalized_obj_space_multifront_hv_gradient = np.zeros((n_mo_obj, n_mo_sol))
        for i_mo_sol in range(0, n_mo_sol):
            w = np.sqrt(np.sum(obj_space_multifront_hv_gradient[:, i_mo_sol] ** 2.0))
            if np.isclose(w, 0):
                w = 1
            if self.obj_space_normalize:
                normalized_obj_space_multifront_hv_gradient[:, i_mo_sol] = obj_space_multifront_hv_gradient[:,
                                                                           i_mo_sol] / w
            else:
                normalized_obj_space_multifront_hv_gradient[:, i_mo_sol] = obj_space_multifront_hv_gradient[:, i_mo_sol]

        dynamic_weights = torch.tensor(normalized_obj_space_multifront_hv_gradient, dtype=torch.float)
        return (dynamic_weights)

def evenly_dist_weights(num_weights, dim):
    return [ret for ret in product(
        np.linspace(0.0, 1.0, num_weights), repeat=dim) if round(sum(ret), 3) == 1.0 and all(r not in (0.0, 1.0) for r in ret)]


class Toy_Hypernetwork(nn.Module):
    def __init__(self, ray_hidden_dim=30, out_dim=1, target_hidden_dim=15, n_hidden=1, n_tasks=2):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_tasks = n_tasks

        self.ray_mlp = nn.Sequential(
            nn.Linear(2, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim)
        )

        for j in range(n_tasks):
            setattr(self, f"task_{j}_weights", nn.Linear(ray_hidden_dim, out_dim))
            setattr(self, f"task_{j}_bias", nn.Linear(ray_hidden_dim, out_dim))

    def forward(self, ray):
        features = self.ray_mlp(ray)

        out_dict = {}

        layer_types = ["task"]
        for i in layer_types:
            if i == "hidden":
                n_layers = self.n_hidden
            elif i == "task":
                n_layers = self.n_tasks

            for j in range(n_layers):
                out_dict[f"{i}{j}.weights"] = getattr(self, f"{i}_{j}_weights")(features)
                out_dict[f"{i}{j}.bias"] = getattr(self, f"{i}_{j}_bias")(features).flatten()

        return out_dict


class Toy_Targetnetwork(nn.Module):
    def __init__(self, out_dim=1, target_hidden_dim=15, n_tasks=2):
        super().__init__()
        self.out_dim = out_dim
        self.n_tasks = n_tasks
        self.target_hidden_dim = target_hidden_dim

    def forward(self, weights=None):
        x = torch.ones([1, 1], dtype=torch.float32).to(device)

        outputs = []
        for j in range(self.n_tasks):
            outputs.append(
                F.linear(
                    x, weight=weights[f'task{j}.weights'].reshape(self.out_dim, x.shape[-1]),
                    bias=weights[f'task{j}.bias']
                )
            )
        return outputs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def toy_loss_1(output):
  return torch.mean(output**2)
def toy_loss_2(output):
  return torch.mean((output-1)**2)

import random
def set_logger():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_logger()

class parse_arg():
  def __init__(self):
    pass
args = parse_arg()
args.no_cuda = False
args.gpus = '0'

def get_device(no_cuda=False, gpus='0'):
    return torch.device(f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu")
device=get_device(no_cuda=args.no_cuda, gpus=args.gpus)

from functions_evaluation import fastNonDominatedSort

import copy
from functions_hv_python3 import HyperVolume

from functions_evaluation import compute_hv_in_higher_dimensions as compute_hv


def evaluate_hv(soluong, start, end, hnet, net, ref_point):
    results = []

    loss1 = toy_loss_1
    loss2 = toy_loss_2
    angles = np.linspace(start, end, soluong, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    rays = np.c_[x, y]
    min_penalty = 999
    for i in range(soluong):
        ray = rays[i]
        ray = torch.from_numpy(
            ray.astype(np.float32).flatten()
        ).to(device)

        weights = hnet(ray)
        output = net(weights)[0]
        l1 = loss1(output)
        l2 = loss2(output)
        loss_batch = torch.stack((l1, l2)).detach().cpu().numpy()
        loss_batch = np.array(loss_batch)
        penalty = np.sum(rays[i].astype(np.float32).flatten() * loss_batch) / (np.sqrt(np.sum(loss_batch ** 2)))
        if penalty < min_penalty:
            min_penalty = penalty
        results.append(loss_batch)

    results = np.array(results, dtype='float32')
    return compute_hv(results.T, ref_point), results, min_penalty


def point_to_line_distance(point, line_point, line_direction):
    numerator = torch.sum((point - line_point) * line_direction)
    denominator = torch.sum(line_direction ** 2)
    t = numerator / denominator

    diff_vector = point - (line_point + t * line_direction)
    distance = torch.sqrt(torch.sum(diff_vector ** 2))
    return distance

def train(device: torch.device, hidden_dim: int, lr: float, wd: float, epochs: int, alpha: float, head: int,
          hesophat: float, cfg):
    training = []
    partition = np.array([1 / head] * head)
    hnet: nn.Module = Toy_Hypernetwork(ray_hidden_dim=hidden_dim, n_tasks=1)
    best_hv = 0
    net_list = []
    for i in range(head):
        net: nn.Module = Toy_Targetnetwork(n_tasks=1)
        net_list.append(net)
    logging.info(f"HN size: {count_parameters(hnet)}")

    hnet = hnet.to(device)
    for net in net_list:
        net = net.to(device)

    loss1 = toy_loss_1
    loss2 = toy_loss_2

    optimizer = torch.optim.Adam(hnet.parameters(), lr=lr, weight_decay=wd)

    ref_point = cfg["ref_point"]
    n_mo_sol = cfg["n_mo_sol"]
    n_mo_obj = cfg["n_mo_obj"]

    start = 0.
    end = np.pi / 2
    print("Start Phase 1")
    phase1_iter = trange(1)

    optimizer_direction = torch.optim.Adam(hnet.parameters(), lr=5e-4, weight_decay=wd)
    for _ in phase1_iter:
        optimizer_direction.zero_grad()
        if (_ + 1) % 50 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= np.sqrt(0.5)
            # patience = 0
            lr *= np.sqrt(0.5)
        penalty_epoch = []

        losses_mean = []
        weights = []
        outputs = []
        rays = []
        penalty = []
        sum_penalty = 0
        for j in range(n_mo_sol):
            random = np.random.uniform(start + j * (end - start) / head, start + (j + 1) * (end - start) / head)
            # random = np.random.uniform(start, end)
            ray = np.array([np.cos(random),
                            np.sin(random)], dtype='float32')

            rays.append(torch.from_numpy(
                ray.flatten()
            ).to(device))

            weights.append(hnet(rays[j]))
            outputs.append(net_list[j](weights[j])[0])
            losses_mean.append(torch.stack([loss1(outputs[j]), loss2(outputs[j])]))

            penalty.append(torch.sum(losses_mean[j] * rays[j]) / (torch.norm(rays[j]) * torch.norm(losses_mean[j])))

            sum_penalty += penalty[j].item()

        direction_loss = 0.
        for phat in penalty[:]:
            direction_loss -= phat
        direction_loss.backward()
        optimizer_direction.step()

        aaa = sum_penalty / float(head)
        phase1_iter.set_description(f"Epochs {_} penalty {aaa:.2f}")

    print("End phase 1 after {}".format(_))
    mo_opt = HvMaximization(n_mo_sol, n_mo_obj, ref_point)

    dem = 0
    epoch_iter = trange(epochs)
    for epoch in epoch_iter:
        dem += 1

        hnet.train()
        optimizer.zero_grad()

        loss_torch_per_sample = []
        loss_numpy_per_sample = []
        loss_per_sample = []
        weights = []
        outputs = []
        rays = []
        penalty = []
        for i in range(n_mo_sol):
            random = np.random.uniform(start, end)
            # random = np.random.uniform(start + i*(end-start)/head, start+ (i+1)*(end-start)/head)
            ray = np.array([np.cos(random),
                            np.sin(random)], dtype='float32')

            rays.append(torch.from_numpy(
                ray.flatten()
            ).to(device))
            weights.append(hnet(rays[i]))

            outputs.append(net_list[i](weights[i])[0])

            loss_per_sample = torch.stack([loss1(outputs[i]), loss2(outputs[i])])

            loss_torch_per_sample.append(loss_per_sample)
            loss_numpy_per_sample.append(loss_per_sample.cpu().detach().numpy())

            line_direction = torch.ones_like(loss_per_sample)
            distance = point_to_line_distance(loss_per_sample, rays[i], line_direction)

            # Penalty term
            penalty.append(distance)

            # penalty.append(torch.sum(loss_torch_per_sample[i] * rays[i]) /
            #                (torch.norm(loss_torch_per_sample[i]) * torch.norm(rays[i])))

        loss_numpy_per_sample = np.array(loss_numpy_per_sample)[np.newaxis, :, :].transpose(0, 2,
                                                                                            1)  # n_samples, obj, sol

        n_samples = 1
        dynamic_weights_per_sample = torch.ones(n_mo_sol, n_mo_obj, n_samples)
        for i_sample in range(0, n_samples):
            weights_task = mo_opt.compute_weights(loss_numpy_per_sample[i_sample, :, :])
            dynamic_weights_per_sample[:, :, i_sample] = weights_task.permute(1, 0)

        dynamic_weights_per_sample = dynamic_weights_per_sample.to(device)
        i_mo_sol = 0
        total_dynamic_loss = torch.mean(torch.sum(dynamic_weights_per_sample[i_mo_sol, :, :]
                                                  * loss_torch_per_sample[i_mo_sol], dim=0))
        for i_mo_sol in range(1, len(net_list)):
            total_dynamic_loss += torch.mean(torch.sum(dynamic_weights_per_sample[i_mo_sol, :, :]
                                                       * loss_torch_per_sample[i_mo_sol], dim=0))

        for idx in range(head):
            total_dynamic_loss += hesophat * penalty[idx] * partition[idx]

        total_dynamic_loss /= head
        total_dynamic_loss.backward()
        optimizer.step()

        penalty = [i.item() for i in penalty]

        if epoch % 300 == 0:
            # hesophat *= 0.5
            training.append([i.detach().cpu().tolist() for i in loss_torch_per_sample])
    return hnet, net, training, partition

cfg = {}
cfg["ref_point"] = (2, 2)
cfg["n_mo_obj"] = 2
cfg["n_mo_sol"] = 16
device = get_device(no_cuda=args.no_cuda, gpus=args.gpus)
hidden_dim=100
lr=1e-3
wd=0.
epochs=10000
alpha=0.2
cfg= cfg
head=cfg["n_mo_sol"]
hesophat=2000.
n_mo_obj = cfg["n_mo_obj"]
n_mo_sol = cfg["n_mo_sol"]
ref_point = cfg["ref_point"]
method = "1_MH"

hnet, net, training, partition = train(device = get_device(no_cuda=args.no_cuda, gpus=args.gpus), hidden_dim=100,
                  lr=1e-3, wd=0., epochs=epochs, alpha=0.2, cfg= cfg, head=cfg['n_mo_sol'], hesophat=hesophat)

results = []
loss1 = toy_loss_1
loss2 = toy_loss_2
soluong = 1000
start = 0.
end = np.pi/2
random_lst = np.linspace(start, end, soluong)
for i in range(soluong):
  random = random_lst[i]
  ray = np.array([np.cos(random),
                      np.sin(random)], dtype='float32')
  ray = torch.from_numpy(
                      ray.flatten()
                  ).to(device)

  weights = hnet(ray)
  output = net(weights)[0]
  l1 = loss1(output)
  l2 = loss2(output)
  results.append([l1, l2])

results = [[i[0].cpu().detach().numpy(), i[1].cpu().detach().numpy()] for i in results]
results = np.array(results, dtype='float32')
plt.scatter(results[:, 0], results[:, 1], s=5)

np.save('results_pro1_hvvs.npy', results)


