# @Time :4/8/2024 3:21 PM
# @Author :Pxmmmm
# @Site :
# @File :flow_TimingGCN.py
# @Version:  0.1


import torch
import numpy as np
import dgl
import torch.nn.functional as F
import random
import pdb
import time
import argparse
import os
from sklearn.metrics import r2_score

from data_manager.data_preprocessing import data_train, data_test
from model.TimingGCN import TimingGCN
from data_manager.data_reporter import *
import matplotlib.pyplot as plt


def test(model):  # at
    model.eval()
    with torch.no_grad():
        def test_dict(data, istrain):
            test_r2_netdelay_batch_sum = 0
            test_r2_celldelay_batch_sum = 0
            test_r2_at_batch_sum = 0
            test_r2_slew_batch_sum = 0
            # test_r2_slack_batch_sum = 0

            test_r2_slack_flatten_batch_sum = 0
            test_r2_slack_unf_batch_sum = 0
            test_r2_slack_endpt_flatten_batch_sum = 0
            test_r2_slack_endpt_unf_batch_sum = 0

            train_r2_netdelay_batch_sum = 0
            train_r2_celldelay_batch_sum = 0
            train_r2_at_batch_sum = 0
            train_r2_slew_batch_sum = 0
            # train_r2_slack_batch_sum = 0

            train_r2_slack_flatten_batch_sum = 0
            train_r2_slack_unf_batch_sum = 0
            train_r2_slack_endpt_flatten_batch_sum = 0
            train_r2_slack_endpt_unf_batch_sum = 0

            for k, (g, ts) in data.items():
                torch.cuda.synchronize()
                time_s = time.time()
                net_delay, cell_delay, at, slew = model(g, ts, groundtruth=groundtruth_test)  # nf
                torch.cuda.synchronize()
                time_t = time.time()

                if torch.isnan(net_delay).any().item() or \
                        torch.isnan(cell_delay).any().item() or \
                        torch.isnan(at).any().item() or \
                        torch.isnan(slew).any().item():
                    print("nan, skip")
                    if (istrain == False):
                        test_r2_netdelay_list.append(0)
                        test_r2_celldelay_list.append(0)
                        test_r2_at_list.append(0)
                        test_r2_slew_list.append(0)

                        test_r2_slack_f_list.append(0)
                        test_r2_slack_unf_list.append(0)
                        test_r2_slack_ep_f_list.append(0)
                        test_r2_slack_ep_unf_list.append(0)
                    else:
                        train_r2_netdelay_list.append(0)
                        train_r2_celldelay_list.append(0)
                        train_r2_at_list.append(0)
                        train_r2_slew_list.append(0)

                        train_r2_slack_f_list.append(0)
                        train_r2_slack_unf_list.append(0)
                        train_r2_slack_ep_f_list.append(0)
                        train_r2_slack_ep_unf_list.append(0)

                    break

                slack = at - g.ndata['n_rats']
                slack
                slack_endpt = slack[g.ndata['n_is_timing_endpt'].bool()]

                # # only compare the input_nodes's net_delay
                # net_delay = net_delay[ts['input_nodes']]
                # truth_net_delay = g.ndata['n_net_delays_log'][ts['input_nodes']]
                truth_net_delay = g.ndata['n_net_delays_log']
                truth_cell_delay = g.edges['cell_out'].data['e_cell_delays']
                truth_at = g.ndata['n_ats']
                truth_slew = g.ndata['n_slew_log']
                truth_slack = truth_at - g.ndata['n_rats']
                truth_slack_endpt = truth_slack[g.ndata['n_is_timing_endpt'].bool()]

                r2_score_netdelay = r2_score(net_delay.cpu().numpy().reshape(-1),
                                             truth_net_delay.cpu().numpy().reshape(-1))
                r2_score_celldelay = r2_score(cell_delay.cpu().numpy().reshape(-1),
                                              truth_cell_delay.cpu().numpy().reshape(-1))
                r2_score_at = r2_score(at.cpu().numpy().reshape(-1), truth_at.cpu().numpy().reshape(-1))
                r2_score_slew = r2_score(slew.cpu().numpy().reshape(-1), truth_slew.cpu().numpy().reshape(-1))

                r2_score_slack_flatten = r2_score(slack.cpu().numpy().reshape(-1),
                                                  truth_slack.cpu().numpy().reshape(-1))
                r2_score_slack_unf = r2_score(slack.cpu().numpy(), truth_slack.cpu().numpy())
                r2_score_slack_endpt_flatten = r2_score(slack_endpt.cpu().numpy().reshape(-1),
                                                        truth_slack_endpt.cpu().numpy().reshape(-1))
                r2_score_slack_endpt_unf = r2_score(slack_endpt.cpu().numpy(), truth_slack_endpt.cpu().numpy())

                if (istrain == False):
                    test_r2_netdelay_batch_sum += r2_score_netdelay
                    test_r2_celldelay_batch_sum += r2_score_celldelay
                    test_r2_at_batch_sum += r2_score_at
                    test_r2_slew_batch_sum += r2_score_slew
                    # test_r2_rat_batch_sum += r2_score_rat

                    test_r2_slack_flatten_batch_sum += r2_score_slack_flatten
                    test_r2_slack_unf_batch_sum += r2_score_slack_unf
                    test_r2_slack_endpt_flatten_batch_sum += r2_score_slack_endpt_flatten
                    test_r2_slack_endpt_unf_batch_sum += r2_score_slack_endpt_unf

                else:
                    train_r2_netdelay_batch_sum += r2_score_netdelay
                    train_r2_celldelay_batch_sum += r2_score_celldelay
                    train_r2_at_batch_sum += r2_score_at
                    train_r2_slew_batch_sum += r2_score_slew
                    # train_r2_rat_batch_sum += r2_score_rat

                    train_r2_slack_flatten_batch_sum += r2_score_slack_flatten
                    train_r2_slack_unf_batch_sum += r2_score_slack_unf
                    train_r2_slack_endpt_flatten_batch_sum += r2_score_slack_endpt_flatten
                    train_r2_slack_endpt_unf_batch_sum += r2_score_slack_endpt_unf

                print(
                    '{:.<12}: time {:.4f} r2_netdelay {:.4f}, r2_celldelay {:.4f}, r2_at {:.4f}, r2_slew {:.4f}, slack_f {:.4f}, slack_unf {:.4f}, slack_ep_f {:.4f}, slack_ep_unf {:.4f}'.format(
                        k,
                        (time_t - time_s) * 1000, r2_score_netdelay, r2_score_celldelay, r2_score_at, r2_score_slew,
                        r2_score_slack_flatten,
                        r2_score_slack_unf,
                        r2_score_slack_endpt_flatten,
                        r2_score_slack_endpt_unf))

            if (istrain == False):
                test_r2_netdelay_list.append(test_r2_netdelay_batch_sum / len(data_test))
                test_r2_celldelay_list.append(test_r2_celldelay_batch_sum / len(data_test))
                test_r2_at_list.append(test_r2_at_batch_sum / len(data_test))
                test_r2_slew_list.append(test_r2_slew_batch_sum / len(data_test))
                # test_r2_rat_list.append(test_r2_rat_batch_sum / len(data_test))

                test_r2_slack_f_list.append(test_r2_slack_flatten_batch_sum / len(data_test))
                test_r2_slack_unf_list.append(test_r2_slack_unf_batch_sum / len(data_test))
                test_r2_slack_ep_f_list.append(test_r2_slack_endpt_flatten_batch_sum / len(data_test))
                test_r2_slack_ep_unf_list.append(test_r2_slack_endpt_unf_batch_sum / len(data_test))

                print(
                    '-----------------test avg----------------- \nr2: netdelay {:1.4f}, celldelay {:1.4f}, at {:1.4f}, slew {:1.4f}, slack_f {:1.4f}, slack_unf {:1.4f}, slack_ep_f {:1.4f}, slack_ep_unf {:1.4f},\n'.format(
                        test_r2_netdelay_batch_sum / len(data_test), test_r2_celldelay_batch_sum / len(data_test),
                        test_r2_at_batch_sum / len(data_test), test_r2_slew_batch_sum / len(data_test),
                        test_r2_slack_flatten_batch_sum / len(data_test),
                        test_r2_slack_unf_batch_sum / len(data_test),
                        test_r2_slack_endpt_flatten_batch_sum / len(data_test),
                        test_r2_slack_endpt_unf_batch_sum / len(data_test),

                    ))
            else:
                train_r2_netdelay_list.append(train_r2_netdelay_batch_sum / len(data_train))
                train_r2_celldelay_list.append(train_r2_celldelay_batch_sum / len(data_train))
                train_r2_at_list.append(train_r2_at_batch_sum / len(data_train))
                train_r2_slew_list.append(train_r2_slew_batch_sum / len(data_train))

                train_r2_slack_f_list.append(train_r2_slack_flatten_batch_sum / len(data_train))
                train_r2_slack_unf_list.append(train_r2_slack_unf_batch_sum / len(data_train))
                train_r2_slack_ep_f_list.append(train_r2_slack_endpt_flatten_batch_sum / len(data_train))
                train_r2_slack_ep_unf_list.append(train_r2_slack_endpt_unf_batch_sum / len(data_train))

                print(
                    '-----------------train avg----------------- \nr2: netdelay {:1.4f}, celldelay {:1.4f}, at {:1.4f}, slew {:1.4f}, slack_f {:1.4f}, slack_unf {:1.4f}, slack_ep_f {:1.4f}, slack_ep_unf {:1.4f},\n'.format(
                        train_r2_netdelay_batch_sum / len(data_train), train_r2_celldelay_batch_sum / len(data_train),
                        train_r2_at_batch_sum / len(data_train), train_r2_slew_batch_sum / len(data_train),
                        # train_r2_rat_batch_sum / len(data_train),

                        train_r2_slack_flatten_batch_sum / len(data_train),
                        train_r2_slack_unf_batch_sum / len(data_train),
                        train_r2_slack_endpt_flatten_batch_sum / len(data_train),
                        train_r2_slack_endpt_unf_batch_sum / len(data_train),

                    ))

        print('======= Training dataset ======')
        test_dict(data_train, True)
        print('======= Test dataset ======')
        test_dict(data_test, False)


def train(model):
    for e in range(epoch):
        model.train()
        train_loss_tot_net_delays, train_loss_tot_cell_delays, train_loss_tot_at, train_loss_tot_slew = 0, 0, 0, 0
        optimizer.zero_grad()

        for k, (g, ts) in random.sample(data_train.items(), batch_size):
            pred_net_delays, pred_cell_delays, pred_at, pred_slew = model(g, ts, groundtruth=groundtruth)

            loss_net_delays = F.mse_loss(pred_net_delays, g.ndata['n_net_delays_log'])
            loss_cell_delays = F.mse_loss(pred_cell_delays, g.edges['cell_out'].data['e_cell_delays'])
            loss_at = F.mse_loss(pred_at, g.ndata['n_ats'])
            loss_slew = F.mse_loss(pred_slew, g.ndata['n_slew_log'])

            train_loss_tot_net_delays += loss_net_delays.item()
            train_loss_tot_cell_delays += loss_cell_delays.item()
            train_loss_tot_at += loss_at.item()
            train_loss_tot_slew += loss_slew.item()

            (loss_net_delays * lambda1 + loss_cell_delays * lambda2 + loss_at * lambda3 + loss_slew).backward()

        optimizer.step()

        if e % 50 == 49:
            with torch.no_grad():
                model.eval()
                test_loss_tot_net_delays, test_loss_tot_cell_delays, test_loss_tot_at, test_loss_tot_slew = 0, 0, 0, 0

                for k, (g, ts) in data_test.items():
                    pred_net_delays, pred_cell_delays, pred_at, pred_slew = model(g, ts, groundtruth=groundtruth)

                    test_loss_tot_net_delays += F.mse_loss(pred_net_delays, g.ndata['n_net_delays_log']).item()
                    test_loss_tot_cell_delays += F.mse_loss(pred_cell_delays,
                                                            g.edges['cell_out'].data['e_cell_delays']).item()
                    test_loss_tot_at += F.mse_loss(pred_at, g.ndata['n_ats']).item()
                    test_loss_tot_slew += F.mse_loss(pred_slew, g.ndata['n_slew_log']).item()

                print(
                    'Epoch={}, net delay {:.6f}/{:.6f}, cell delay {:.6f}/{:.6f}, at {:.6f}/{:.6f}, slew {:.6f}/{:.6f}'.format(
                        e,
                        train_loss_tot_net_delays / batch_size,
                        test_loss_tot_net_delays / len(data_test),

                        train_loss_tot_cell_delays / batch_size,
                        test_loss_tot_cell_delays / len(data_test),

                        train_loss_tot_at / batch_size,
                        test_loss_tot_at / len(data_test),

                        train_loss_tot_slew / batch_size,
                        test_loss_tot_slew / len(data_test)
                    ))

                # append loss list
                train_loss_netdelay_list.append(train_loss_tot_net_delays / batch_size)
                eval_loss_netdelay_list.append(test_loss_tot_net_delays / len(data_test))

                train_loss_celldelay_list.append(train_loss_tot_cell_delays / batch_size)
                eval_loss_celldelay_list.append(test_loss_tot_cell_delays / len(data_test))

                train_loss_at_list.append(train_loss_tot_at / batch_size)
                eval_loss_at_list.append(test_loss_tot_at / len(data_test))

                train_loss_slew_list.append(train_loss_tot_slew / batch_size)
                eval_loss_slew_list.append(test_loss_tot_slew / len(data_test))

            if e % 200 == 199:  # or (e > 6000 and test_loss_tot_ats_prop / len(data_test) < 6):
                if checkpoint:
                    save_path = './checkpoints/{}/{}.pth'.format(checkpoint, e)
                    torch.save(model.state_dict(), save_path)
                    print('saved model to', save_path)
                try:
                    test(model)

                except ValueError as e:
                    print(e)
                    print('Error testing, but ignored')


lambda1 = 1
lambda2 = 1
lambda3 = 1
epoch = 25000  # 18000
lr = 0.0004
batch_size = 7  # 1  # 7

model = TimingGCN()
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

groundtruth = True  # False  # False  # False  # True
groundtruth_test = False  # True  # False  # True  # False  # True  # False  # True  # False  # True  # False  # True  # False
load_param_pretrain_path = False  # 7999  # False  # 599#7199  # False  # 1999  # 19599  # 9799  # 3999  # 3999  # False
load_param_pretrain_checkpoint = "0425_TimingGCN_bechmark2_e8000_rdgly_lr3_gt1_load1_2"
checkpoint = "08_atcd_specul"
load_path = 15799  # 22599  # False  # 24999  # False  # 7999#False  # 599
# checkpoint = "0426_TimingGCN_bm2_e25000_rdgly_lr4_gt1_0"
# checkpoint = "0426_TimingGCN_bm2_e25000_rdgly_lr4_gt1_0"
# checkpoint = "0426_TimingGCN_bm2_e25000_rdgly_lr4_gt1_0"
loop_gt0_result = False  # True  # False  # True  # False  # True  # False  # True  # False  # True  # False  # True  # False  # True  # False  # True  # False  # True  # False  # True  # False  # True

if loop_gt0_result:
    checkpoint_gt0 = checkpoint + '_loopgt0'

    for i in range(int(epoch / 200)):
        path = i * 200 + 199
        print("---------------------{}--------------------".format(path))
        model.load_state_dict(torch.load('./checkpoints/{}/{}.pth'.format(checkpoint, path)))
        test(model)
    # report_loss(checkpoint_gt0, isSave=True)
    report_r2(checkpoint_gt0, isSave=True, isPlot=False)
    report_slack(checkpoint_gt0, isSave=True, isSingle=False, isPlot=False)

elif load_path:
    assert checkpoint, 'no checkpoints dir specified'
    model.load_state_dict(torch.load('./checkpoints/{}/{}.pth'.format(checkpoint, load_path)))
    test(model)


else:
    if checkpoint:
        print('saving logs and models to ./checkpoints/{}'.format(checkpoint))
        os.makedirs('./checkpoints/{}'.format(checkpoint))  # exist not ok
        # stdout_f = './checkpoints/{}/stdout.log'.format(checkpoint)
        # stderr_f = './checkpoints/{}/stderr.log'.format(checkpoint)
        # with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
        if load_param_pretrain_path:
            print("load state dict")
            model.load_state_dict(
                torch.load('./checkpoints/{}/{}.pth'.format(load_param_pretrain_checkpoint, load_param_pretrain_path)))

        test(model)
        train(model)
        report_loss(checkpoint, isSave=True)
        report_r2(checkpoint, isSave=True)
        report_slack(checkpoint, isSave=True, isSingle=False)

    else:
        print('No checkpoints is specified. abandoning all model checkpoints and logs')
        train(model)
        report_loss(checkpoint, isSave=True)
        report_r2(checkpoint, isSave=True)
        report_slack(checkpoint, isSave=True)
