from tqdm import trange
from collections import defaultdict
import torch
import numpy as np
import torch.nn.functional as F
from pymoo.factory import get_performance_indicator

import sys

from hvtest import get_best_voronoi, sample_n_points_from_voronoi
from phn.solvers import MultiHead

import argparse
import json
from pathlib import Path

from tqdm import trange
from collections import defaultdict
import logging

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from utils import (
    set_seed,
    set_logger,
    count_parameters,
    get_device,
    save_args,
)
from models import HyperNet, TargetNet

from load_data import Dataset

from phn.solvers import MultiHead
from scipy.spatial import Delaunay
from modules.functions_evaluation import compute_hv_in_higher_dimensions as compute_hv
from pymoo.factory import get_reference_directions
def get_test_rays():
    test_rays = get_reference_directions("das-dennis", 4, n_partitions=10).astype(
        np.float32
    )
    return test_rays


@torch.no_grad()
def evaluate(hypernet, targetnet, loader, rays, device,epoch,name,n_tasks):
    hypernet.eval()
    results = defaultdict(list)
    loss_total = None
    #front = []
    for ray in rays:
        ray = torch.from_numpy(ray.astype(np.float32)).to(device)
        # ray = ray / np.sqrt(np.sum(ray ** 2))

        ray /= ray.sum()

        total = 0.0
        full_losses = []
        for batch in loader:
            hypernet.zero_grad()

            batch = (t.to(device) for t in batch)
            xs, ys = batch
            bs = len(ys)

            weights = hypernet(ray)
            pred = targetnet(xs, weights)

            # loss
            curr_losses = get_losses(pred, ys)

            ray = ray.squeeze(0)

            # losses
            full_losses.append(curr_losses.detach().cpu().numpy())
            total += bs
        if loss_total is None:
            loss_total = np.array(np.array(full_losses).mean(0).tolist(),dtype='float32')
        else:
            loss_total += np.array(np.array(full_losses).mean(0).tolist(),dtype='float32')
        results["ray"].append(ray.cpu().numpy().tolist())
        results["loss"].append(np.array(full_losses).mean(0).tolist())
    print("\n")
    # print(str(name)+" losses at "+str(epoch)+":",loss_total/len(rays))
    hv = get_performance_indicator(
        "hv",
        ref_point=np.ones(
            n_tasks,
        ),
    )
    hv_result = hv.do(np.array(results["loss"]))
    results["hv"] = hv_result
    results["loss"] = np.array(results["loss"], dtype='float32')
    # results["hv"] = compute_hv(results["loss"].T, ref_point)
    return results

    # return results

def point_to_line_distance(point, line_point, line_direction):
    numerator = torch.sum((point - line_point) * line_direction)
    denominator = torch.sum(line_direction ** 2)
    t = numerator / denominator

    diff_vector = point - (line_point + t * line_direction)
    distance = torch.sqrt(torch.sum(diff_vector ** 2))
    return distance

# ---------
# Task loss
# ---------
def get_losses(pred, label):
    return F.mse_loss(pred, label, reduction="none").mean(0)

def train_MSH(solver,
    hnet,
    net,
    optimizer,
    optimizer_direction,
    device,
    n_mo_sol,
    n_tasks,
    head,
    train_loader,
    val_loader,
    test_loader,
    epochs,
    test_rays,
    lamda,
    lr,best_voronoi_cells) -> None:
    #print(n_mo_sol)
    net_list = []
    for i in range(head):
        net_list.append(net)
    print("Start Phase 1")
    phase1_iter = trange(1)
    for _ in phase1_iter:
        penalty_epoch = []
        for i, batch in enumerate(train_loader):
            hnet.train()
            optimizer_direction.zero_grad()

            x, y = batch
            x = x.to(device)
            y = y.to(device)

            losses_mean= []
            weights = []
            outputs = []
            rays = []
            penalty = []
            sum_penalty = 0

            for j in range(n_mo_sol):
                ray = np.random.dirichlet([1/n_tasks]*n_tasks, 1).astype(np.float32)
                rays.append(torch.from_numpy(
                                ray.flatten()
                            ).to(device))
                weights.append(hnet(rays[j]))
                #print(weights)
                outputs.append(net_list[j](x, weights[j]))
                losses_mean.append(torch.stack([F.mse_loss(outputs[j][:, k], y[:, k]) for k in range(n_tasks)
                                                ]))
                penalty.append(torch.sum(losses_mean[j]*rays[j])/(torch.norm(rays[j])*torch.norm(losses_mean[j])))
                sum_penalty += penalty[-1].item()
            penalty_epoch.append(sum_penalty/float(head))
            
            direction_loss = penalty[0]
            for phat in penalty[:]:
                direction_loss -= phat
            direction_loss.backward()
            optimizer_direction.step()
        print("Epochs {} penalty{:.2f}".format(_, np.mean(np.array(penalty_epoch))))
    print("End phase 1")

    print("Start Phase 2")

    epoch_iter = trange(epochs)
    count = 0
    patience = 0
    early_stop = 0
    test_hv = -1
    val_hv = -1
    min_loss = [999]*n_tasks
    training_loss = []
    for epoch in epoch_iter:
        count += 1
        
        if early_stop == 100:
            print("Early stop.")
            break
        
        if (patience+1) % 40 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= np.sqrt(0.5)
            patience = 0
            lr *= np.sqrt(0.5)
            # lamda*=np.sqrt(0.5)
            print(param_group['lr'])

        for i, batch in enumerate(train_loader):
            hnet.train()
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            loss_torch = []
            loss_numpy = []
            weights = []
            outputs = []
            rays = []
            penalty = []

            sum_penalty = 0
            training_loss_epoch = []
            sampled_ray = sample_n_points_from_voronoi(best_voronoi_cells, n_mo_sol)
            for j in range(n_mo_sol):       

                ray = sampled_ray[j]

                ray = ray.astype(np.float32)
                rays.append(torch.from_numpy(
                                ray.flatten()
                            ).to(device))
                weights.append(hnet(rays[j]))
                outputs.append(net_list[j](x, weights[j]))
                loss_per_sample = torch.stack([F.mse_loss(outputs[j][:, k], y[:, k]) for k in range(n_tasks)])
                loss_torch.append(torch.stack([F.mse_loss(outputs[j][:, k], y[:, k]) for k in range(n_tasks)]))
                loss_numpy.append([idx.cpu().detach().numpy() for idx in loss_torch[j]])
                line_direction = torch.ones_like(loss_per_sample)

                distance = point_to_line_distance(loss_per_sample, rays[j], line_direction)

                # Penalty term
                penalty.append(distance)
                # penalty.append((torch.sum(loss_torch[j]*rays[j]))/(torch.norm(loss_torch[j])*
                #                                 torch.norm(rays[j])))
                sum_penalty += penalty[-1].item()
            # print(penalty)

            loss_numpy = np.array(loss_numpy).T
            loss_numpy = loss_numpy[np.newaxis, :, :]
            training_loss_epoch.append(loss_numpy)
            total_dynamic_loss = solver.get_weighted_loss(loss_numpy,device,loss_torch,head,penalty,lamda,1)
            total_dynamic_loss.backward()
            optimizer.step()
        training_loss_epoch = np.mean(np.stack(training_loss_epoch), axis=0)
        training_loss.append(training_loss_epoch)
        for _ in range(n_tasks):
            min_loss[_] = min(min_loss[_], np.min(training_loss_epoch[0, _]))
        
        last_eval = epoch

        val_epoch_results = evaluate(
            hypernet=hnet,
            targetnet=net,
            loader=val_loader,
            rays=test_rays,
            device=device,
            name = "Val",
            epoch = epoch,
            n_tasks = n_tasks,
        )

        if val_epoch_results["hv"] > val_hv:
            val_hv = val_epoch_results["hv"]
            print(val_hv)

            torch.save(hnet,'save_models/Jura_MH_Freely1_'+ str(head) + "_" + str(lamda) + '_best.pt')

            patience = 0
        else:
            patience += 1
            early_stop += 1
        # print("Epoch", epoch, 'val_hv', val_hv)

    print("End Phase 2")

    # hnet = torch.load('/root/cmm/mhe/Jura/save_models/Jura_MH_Freely1_'+ str(head) + "_" + str(lamda) + '_best.pt')
    # test_rays = sample_n_points_from_voronoi(best_voronoi_cells, 286)
    test_epoch_results = evaluate(
        hypernet=hnet,
        targetnet=net,
        loader=test_loader,
        rays=test_rays,
        device=device,
        name = "Test",
        epoch = 1,
        n_tasks = n_tasks,
    )
    pf = np.array(test_epoch_results['loss'])
    # print(pf)
    with open('output_hvd.txt', 'w') as f:
        np.savetxt(f, pf)

    np.save("fronts/Jura_MH_Freely1_"+ str(head) + "_" + str(lamda) + "_front.npy", pf)

    print("HV on test:",test_epoch_results)
    print("Best HV on val:",val_hv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jura")

    parser.add_argument("--datapath", type=str, default="data/jura.arff", help="path to data")
    parser.add_argument("--n-epochs", type=int, default=1000, help="num. epochs")

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="train on gpu"
    )
    parser.add_argument("--gpus", type=str, default="5", help="gpu device")
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="optimizer type",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")

    parser.add_argument("--seed", type=int, default=42, help="random seed")


    parser.add_argument("--n-mo-sol", type=int, default=8, help="random seed")

    parser.add_argument("--lamda", type=float, default=0.001, help="penalty parameter")

    args = parser.parse_args()
    

    set_seed(args.seed)
    set_logger()

    n_mo_obj = 4
    ref_point = [1]*n_mo_obj
    n_tasks = n_mo_obj
    head = args.n_mo_sol
    device=get_device(no_cuda=args.no_cuda, gpus=args.gpus)

    solver = MultiHead(args.n_mo_sol, n_mo_obj, ref_point)
    hnet: nn.Module = HyperNet()
    net: nn.Module = TargetNet()
    hnet = hnet.to(device)
    net = net.to(device)
    optimizer = torch.optim.Adam(hnet.parameters(), lr=args.lr, weight_decay=0.)
    optimizer_direction = torch.optim.Adam(hnet.parameters(), lr=1e-4, weight_decay=0.)

    train_set, val_set, test_set = Dataset(args.datapath).get_data()

    bs = args.batch_size

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=bs, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=bs, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=bs, shuffle=False, num_workers=0
    )


    # test_rays = get_test_rays()
    lamda = args.lamda
    # sys.stdout = open("log/Jura_MH_Freely1_"+ str(head) + "_" + str(lamda) + "_" + str(bs) + "_log.txt", 'w')
    m, n, d = 10, args.n_mo_sol, n_mo_obj
    best_voronoi_cells = get_best_voronoi(m, n, d, num_generations=100)
    test_rays = sample_n_points_from_voronoi(best_voronoi_cells, 286)
    train_MSH(solver,
        hnet,
        net,
        optimizer,
        optimizer_direction,
        device,
        args.n_mo_sol,
        n_tasks,
        head,
        train_loader,
        val_loader,
        test_loader,
        args.n_epochs,
        test_rays,
        lamda,
        args.lr,
        best_voronoi_cells
        )
    # save_args(folder=args.out_dir, args=args)
    sys.stdout.close()