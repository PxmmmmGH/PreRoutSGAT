# @Time :4/20/2024 10:51 AM
# @Author :Pxmmmm
# @File :data_reporter.py
# @Version:  0.1
# @Conten: data processing


import matplotlib.pyplot as plt
import torch

import numpy as np

train_r2_netdelay_list = []
test_r2_netdelay_list = []

train_r2_celldelay_list = []
test_r2_celldelay_list = []

train_r2_at_list = []
test_r2_at_list = []

train_r2_slew_list = []
test_r2_slew_list = []

train_r2_rat_list = []
test_r2_rat_list = []

train_r2_slack_list = []
test_r2_slack_list = []

train_r2_slack_f_list = []
test_r2_slack_f_list = []

train_r2_slack_unf_list = []
test_r2_slack_unf_list = []

train_r2_slack_ep_f_list = []
test_r2_slack_ep_f_list = []

train_r2_slack_ep_unf_list = []
test_r2_slack_ep_unf_list = []

########################################

train_loss_netdelay_list = []
eval_loss_netdelay_list = []

train_loss_celldelay_list = []
eval_loss_celldelay_list = []

train_loss_at_list = []
eval_loss_at_list = []

train_loss_slew_list = []
eval_loss_slew_list = []

train_loss_rat_list = []
eval_loss_rat_list = []

train_loss_slack_list = []
eval_loss_slack_list = []


def report_loss(checkpoint, isSave=False, isRat=False):
    print("train_loss_netdelay_list", train_loss_netdelay_list)
    print("test_loss_netdelay_list", eval_loss_netdelay_list)
    print("train_loss_celldelay_list", train_loss_celldelay_list)
    print("test_loss_celldelay_list", eval_loss_celldelay_list)
    print("train_loss_at_list", train_loss_at_list)
    print("test_loss_at_list", eval_loss_at_list)
    print("train_loss_slew_list", train_loss_slew_list)
    print("test_loss_slew_list", eval_loss_slew_list)
    print("train_loss_rat_list", train_loss_rat_list)
    print("test_loss_rat_list", eval_loss_rat_list)
    # print("train_loss_rat_list", train_loss_rat_list)
    # print("test_loss_rat_list", test_loss_rat_list)

    num_loss_list = range(0, len(train_loss_netdelay_list))
    plt.plot(num_loss_list, train_loss_netdelay_list, label='train_loss_netdelay_list')
    plt.plot(num_loss_list, eval_loss_netdelay_list, label='test_loss_netdelay_list')
    plt.plot(num_loss_list, train_loss_celldelay_list, label='train_loss_celldelay_list')
    plt.plot(num_loss_list, eval_loss_celldelay_list, label='test_loss_celldelay_list')

    plt.title('loss_netdelay_cell_delay')
    plt.legend()
    plt.xlabel('time/50epoch')
    plt.ylabel('mse loss')
    plt.ylim(0, 0.5)
    plt.xlim(0, num_loss_list[-1] + 1)
    if isSave:
        plt.savefig('./data/pic/{}_loss_netdelay_cell_delay.png'.format(checkpoint))
    plt.show()

    plt.plot(num_loss_list, train_loss_at_list, label='train_loss_at_list')
    plt.plot(num_loss_list, eval_loss_at_list, label='test_loss_at_list')
    plt.title('loss_at')
    plt.legend()
    plt.xlabel('time/50epoch')
    plt.ylabel('mse loss')
    plt.ylim(0, 0.5)
    plt.xlim(5, num_loss_list[-1] + 1)
    if isSave:
        plt.savefig('./data/pic/{}_loss_arrival_time.png'.format(checkpoint))
    plt.show()

    plt.plot(num_loss_list, train_loss_slew_list, label='train_loss_slew_list')
    plt.plot(num_loss_list, eval_loss_slew_list, label='test_loss_slew_list')
    plt.title('loss_slew')
    plt.legend()
    plt.xlabel('time/50epoch')
    plt.ylabel('mse loss')
    plt.ylim(0, 0.5)
    plt.xlim(5, num_loss_list[-1] + 1)
    if isSave:
        plt.savefig('./data/pic/{}_loss_arrival_time.png'.format(checkpoint))
    plt.show()

    if isRat:
        plt.plot(num_loss_list, train_loss_rat_list, label='train_loss_rat_list')
        plt.plot(num_loss_list, eval_loss_rat_list, label='test_loss_rat_list')
        plt.title('loss_rat')
        plt.legend()
        plt.xlabel('time/50epoch')
        plt.ylabel('mse loss')
        plt.ylim(0, 0.5)
        plt.xlim(5, num_loss_list[-1] + 1)
        if isSave:
            plt.savefig('./data/pic/{}_loss_rat.png'.format(checkpoint))
        plt.show()

    # loss_diff_netdelay = [train_loss_netdelay_list[i] - eval_loss_netdelay_list[i] for i in num_loss_list]
    # loss_diff_celldelay = [train_loss_celldelay_list[i] - eval_loss_celldelay_list[i] for i in num_loss_list]
    # loss_diff_at = [train_loss_at_list[i] - eval_loss_at_list[i] for i in num_loss_list]
    # loss_diff_slew = [train_loss_slew_list[i] - eval_loss_slew_list[i] for i in num_loss_list]
    # # loss_diff_rat = [train_loss_rat_list[i] - test_loss_rat_list[i] for i in num_loss_list]
    #
    # plt.plot(num_loss_list, loss_diff_netdelay, label='loss_diff_netdelay')
    # plt.plot(num_loss_list, loss_diff_celldelay, label='loss_diff_celldelay')
    # plt.plot(num_loss_list, loss_diff_at, label='loss_diff_at')
    # plt.plot(num_loss_list, loss_diff_slew, label='loss_diff_slew')
    # plt.plot(num_loss_list, loss_diff_rat, label='loss_diff_rat')
    #
    # plt.title('loss_diff')
    # plt.legend()
    # plt.xlabel('time/50epoch')
    # plt.ylabel('mse loss')
    # plt.ylim(-1, 1)
    # plt.xlim(0, num_loss_list[-1] + 1)
    # if isSave:
    #     plt.savefig('./data/pic/{}_loss_diff.png'.format(checkpoint))
    # plt.show()

    if isSave:
        np.savetxt('./data/datalist/{}_train_loss_netdelay_list.txt'.format(checkpoint), train_loss_netdelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_test_loss_netdelay_list.txt'.format(checkpoint), eval_loss_netdelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_train_loss_celldelay_list.txt'.format(checkpoint), train_loss_celldelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_test_loss_celldelay_list.txt'.format(checkpoint), eval_loss_celldelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_train_loss_at_list.txt'.format(checkpoint), train_loss_at_list, delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_test_loss_at_list.txt'.format(checkpoint), eval_loss_at_list, delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_train_loss_slew_list.txt'.format(checkpoint), train_loss_slew_list,
                   delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_test_loss_slew_list.txt'.format(checkpoint), eval_loss_slew_list, delimiter=' ',
                   fmt='%f')

        # # np.savetxt('./data/datalist/{}_test_loss_rat_list.txt'.format(checkpoint), test_loss_rat_list, delimiter=' ', fmt='%f')
        # np.savetxt('./data/datalist/{}_loss_diff_netdelay.txt'.format(checkpoint), loss_diff_netdelay,
        #            delimiter=' ', fmt='%f')
        # np.savetxt('./data/datalist/{}_loss_diff_celldelay.txt'.format(checkpoint), loss_diff_celldelay, delimiter=' ',
        #            fmt='%f')
        # np.savetxt('./data/datalist/{}_loss_diff_at.txt'.format(checkpoint), loss_diff_at, delimiter=' ',
        #            fmt='%f')
        # np.savetxt('./data/datalist/{}_loss_diff_slew.txt'.format(checkpoint), loss_diff_slew, delimiter=' ',
        #            fmt='%f')
        # # np.savetxt('./data/datalist/{}_loss_diff_rat.txt'.format(checkpoint), loss_diff_rat, delimiter=' ', fmt='%f')


def report_r2(checkpoint, isSave=False, isRat=False, isPlot=True):
    print("test_r2_netdelay_list", test_r2_netdelay_list)
    print("test_r2_celldelay_list", test_r2_celldelay_list)
    print("test_r2_at_list", test_r2_at_list)
    print("test_r2_slew_list", test_r2_slew_list)

    print("train_r2_netdelay_list", train_r2_netdelay_list)
    print("train_r2_celldelay_list", train_r2_celldelay_list)
    print("train_r2_at_list", train_r2_at_list)
    print("train_r2_slew_list", train_r2_slew_list)

    if isRat:
        print("train_r2_rat_list", train_r2_rat_list)
        print("test_r2_rat_list", test_r2_rat_list)

    num_r2 = range(0, len(train_r2_netdelay_list))

    if isPlot:
        plt.plot(num_r2, test_r2_netdelay_list, label='test_r2_netdelay')
        plt.plot(num_r2, test_r2_celldelay_list, label='test_r2_celldelay')
        plt.plot(num_r2, test_r2_at_list, label='test_r2_at_list')
        plt.plot(num_r2, test_r2_slew_list, label='test_r2_slew_list')
        if isRat:
            plt.plot(num_r2, test_r2_rat_list, label='test_r2_rat_list')

        plt.title('test R2 history')
        plt.legend()
        plt.xlabel('time/200epoch')
        plt.ylabel('r2score')
        plt.ylim(0.5, 1)
        plt.xlim(5, num_r2[-1] + 1)
        if isSave:
            plt.savefig('./data/pic/{}_test_r2.png'.format(checkpoint))
        plt.show()

        plt.plot(num_r2, train_r2_netdelay_list, label='train_r2_netdelay')
        plt.plot(num_r2, train_r2_celldelay_list, label='train_r2_celldelay')
        plt.plot(num_r2, train_r2_at_list, label='train_r2_at_list')
        plt.plot(num_r2, train_r2_slew_list, label='train_r2_slew_list')
        if isRat:
            plt.plot(num_r2, train_r2_rat_list, label='test_r2_rat_list')

        plt.title('train R2 history')
        plt.legend()
        plt.xlabel('time/200epoch')
        plt.ylabel('r2score')
        plt.ylim(0.5, 1)
        plt.xlim(5, num_r2[-1] + 1)
        if isSave:
            plt.savefig('./data/pic/{}_train_r2.png'.format(checkpoint))
        plt.show()

    if isSave:
        np.savetxt('./data/datalist/{}_test_r2_netdelay_list.txt'.format(checkpoint), test_r2_netdelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_test_r2_celldelay_list.txt'.format(checkpoint), test_r2_celldelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_test_r2_at_list.txt'.format(checkpoint), test_r2_at_list, delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_test_r2_slew_list.txt'.format(checkpoint), test_r2_slew_list, delimiter=' ',
                   fmt='%f')

        np.savetxt('./data/datalist/{}_train_r2_netdelay_list.txt'.format(checkpoint), train_r2_netdelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_train_r2_celldelay_list.txt'.format(checkpoint), train_r2_celldelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_train_r2_at_list.txt'.format(checkpoint), train_r2_at_list, delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_train_r2_slew_list.txt'.format(checkpoint), train_r2_slew_list, delimiter=' ',
                   fmt='%f')

        if isRat:
            np.savetxt('./data/datalist/{}_test_r2_rat_list.txt'.format(checkpoint), test_r2_rat_list, delimiter=' ',
                       fmt='%f')
            np.savetxt('./data/datalist/{}_train_r2_rat_list.txt'.format(checkpoint), train_r2_rat_list, delimiter=' ',
                       fmt='%f')


def report_slack(checkpoint, isSave=False, isSingle=True, isPlot=True):
    print("test_r2_at_list", test_r2_at_list)
    print("train_r2_at_list", train_r2_at_list)

    num_r2 = range(0, len(test_r2_at_list))

    if isPlot:
        plt.plot(num_r2, test_r2_at_list, label='test_r2_at_list')
        plt.plot(num_r2, train_r2_at_list, label='train_r2_at_list')

        plt.title('at R2 history')
        plt.legend()
        plt.xlabel('time/200epoch')
        plt.ylabel('r2score')
        plt.ylim(0.5, 1)
        plt.xlim(5, num_r2[-1] + 1)
        if isSave:
            plt.savefig('./data/pic/{}_at_r2.png'.format(checkpoint))

        plt.show()

    if isSave:
        np.savetxt('./data/datalist/{}_test_r2_at_list.txt'.format(checkpoint), test_r2_at_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_train_r2_at_list.txt'.format(checkpoint), train_r2_at_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_test_r2_slack_list.txt'.format(checkpoint), test_r2_slack_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_train_r2_slack_list.txt'.format(checkpoint), train_r2_slack_list,
                   delimiter=' ', fmt='%f')
    if isPlot:
        if isSingle:
            print("test_r2_slack_list", test_r2_slack_list)
            print("train_r2_slack_list", train_r2_slack_list)

            num_r2 = range(0, len(test_r2_slack_list))
            plt.plot(num_r2, test_r2_slack_list, label='test_r2_slack_list')
            plt.plot(num_r2, train_r2_slack_list, label='train_r2_slack_list')

            plt.title('slack R2 history')
            plt.legend()
            plt.xlabel('time/200epoch')
            plt.ylabel('r2score')
            plt.ylim(0.5, 1)
            plt.xlim(5, num_r2[-1] + 1)
            if isSave:
                plt.savefig('./data/pic/{}_slack_r2.png'.format(checkpoint))
            plt.show()

        else:
            num_r2 = range(0, len(test_r2_slack_f_list))

            plt.plot(num_r2, test_r2_slack_f_list, label='test_r2_slack_f_list')
            plt.plot(num_r2, test_r2_slack_unf_list, label='test_r2_slack_unf_list')
            plt.plot(num_r2, test_r2_slack_ep_f_list, label='test_r2_slack_ep_f_list')
            plt.plot(num_r2, test_r2_slack_ep_unf_list, label='test_r2_slack_ep_unf_list')

            plt.title('test slack R2 history')
            plt.legend()
            plt.xlabel('time/200epoch')
            plt.ylabel('r2score')
            plt.ylim(0.5, 1)
            plt.xlim(5, num_r2[-1] + 1)

            if isSave:
                plt.savefig('./data/pic/{}_test_slack_r2.png'.format(checkpoint))
            plt.show()

            plt.plot(num_r2, train_r2_slack_f_list, label='train_r2_slack_f_list')
            plt.plot(num_r2, train_r2_slack_unf_list, label='train_r2_slack_unf_list')
            plt.plot(num_r2, train_r2_slack_ep_f_list, label='train_r2_slack_ep_f_list')
            plt.plot(num_r2, train_r2_slack_ep_unf_list, label='train_r2_slack_ep_unf_list')

            plt.title('train slack R2 history')
            plt.legend()
            plt.xlabel('time/200epoch')
            plt.ylabel('r2score')
            plt.ylim(0.5, 1)
            plt.xlim(5, num_r2[-1] + 1)

            if isSave:
                plt.savefig('./data/pic/{}_train_slack_r2.png'.format(checkpoint))
            plt.show()

    if isSave:
        np.savetxt('./data/datalist/{}_test_r2_slack_f_list.txt'.format(checkpoint), test_r2_slack_f_list,
                   delimiter=' ', fmt='%f')

        np.savetxt('./data/datalist/{}_test_r2_slack_unf_list.txt'.format(checkpoint), test_r2_slack_unf_list,
                   delimiter=' ', fmt='%f')

        np.savetxt("./data/datalist/{}_test_r2_slack_ep_f_list.txt".format(checkpoint), test_r2_slack_ep_f_list,
                   delimiter=' ', fmt='%f')

        np.savetxt('./data/datalist/{}_test_r2_slack_ep_unf_list.txt'.format(checkpoint),
                   test_r2_slack_ep_unf_list,
                   delimiter=' ', fmt='%f')

        np.savetxt('./data/datalist/{}_train_r2_slack_f_list.txt'.format(checkpoint), train_r2_slack_f_list,
                   delimiter=' ', fmt='%f')

        np.savetxt('./data/datalist/{}_train_r2_slack_unf_list.txt'.format(checkpoint), train_r2_slack_unf_list,
                   delimiter=' ', fmt='%f')

        np.savetxt('./data/datalist/{}_train_r2_slack_ep_f_list.txt'.format(checkpoint),
                   train_r2_slack_ep_f_list,
                   delimiter=' ', fmt='%f')

        np.savetxt('./data/datalist/{}_train_r2_slack_ep_unf_list.txt'.format(checkpoint),
                   train_r2_slack_ep_unf_list,
                   delimiter=' ', fmt='%f')


def report_loss_withrat(checkpoint, isSave=False):
    print("train_loss_netdelay_list", train_loss_netdelay_list)
    print("test_loss_netdelay_list", eval_loss_netdelay_list)
    print("train_loss_celldelay_list", train_loss_celldelay_list)
    print("test_loss_celldelay_list", eval_loss_celldelay_list)
    print("train_loss_at_list", train_loss_at_list)
    print("test_loss_at_list", eval_loss_at_list)
    print("train_loss_slew_list", train_loss_slew_list)
    print("test_loss_slew_list", eval_loss_slew_list)
    print("train_loss_rat_list", train_loss_rat_list)
    print("test_loss_rat_list", eval_loss_rat_list)

    num_loss_list = range(0, len(train_loss_netdelay_list))
    plt.plot(num_loss_list, train_loss_netdelay_list, label='train_loss_netdelay_list')
    plt.plot(num_loss_list, train_loss_celldelay_list, label='train_loss_celldelay_list')
    plt.plot(num_loss_list, train_loss_slew_list, label='train_loss_slew_list')

    plt.title('loss_netdelay_celldelay_slew')
    plt.legend()
    plt.xlabel('time/50epoch')
    plt.ylabel('mse loss')
    plt.ylim(0, 0.5)
    plt.xlim(0, num_loss_list[-1] + 1)
    if isSave:
        plt.savefig('./data/pic/{}_loss_netdelay_celldelay_slew.png'.format(checkpoint))
    plt.show()

    plt.plot(num_loss_list, train_loss_at_list, label='train_loss_at_list')
    plt.plot(num_loss_list, train_loss_rat_list, label='train_loss_rat_list')
    plt.title('loss_at')
    plt.legend()
    plt.xlabel('time/50epoch')
    plt.ylabel('mse loss')
    plt.ylim(0, 0.5)
    plt.xlim(5, num_loss_list[-1] + 1)
    if isSave:
        plt.savefig('./data/pic/{}_loss_at_rat.png'.format(checkpoint))
    plt.show()

    if isSave:
        np.savetxt('./data/datalist/{}_train_loss_netdelay_list.txt'.format(checkpoint), train_loss_netdelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_test_loss_netdelay_list.txt'.format(checkpoint), eval_loss_netdelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_train_loss_celldelay_list.txt'.format(checkpoint), train_loss_celldelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_test_loss_celldelay_list.txt'.format(checkpoint), eval_loss_celldelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_train_loss_at_list.txt'.format(checkpoint), train_loss_at_list, delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_test_loss_at_list.txt'.format(checkpoint), eval_loss_at_list, delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_train_loss_slew_list.txt'.format(checkpoint), train_loss_slew_list,
                   delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_test_loss_slew_list.txt'.format(checkpoint), eval_loss_slew_list, delimiter=' ',
                   fmt='%f')

        np.savetxt('./data/datalist/{}_train_loss_rat_list.txt'.format(checkpoint), train_loss_rat_list,
                   delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_test_loss_rat_list.txt'.format(checkpoint), eval_loss_rat_list, delimiter=' ',
                   fmt='%f')


def report_r2_withrat(checkpoint, isSave=False):
    print("test_r2_netdelay_list", test_r2_netdelay_list)
    print("test_r2_celldelay_list", test_r2_celldelay_list)
    print("test_r2_at_list", test_r2_at_list)
    print("test_r2_slew_list", test_r2_slew_list)
    print("test_r2_rat_list", test_r2_rat_list)

    print("train_r2_netdelay_list", train_r2_netdelay_list)
    print("train_r2_celldelay_list", train_r2_celldelay_list)
    print("train_r2_at_list", train_r2_at_list)
    print("train_r2_slew_list", train_r2_slew_list)
    print("train_r2_rat_list", train_r2_rat_list)

    num_r2 = range(0, len(train_r2_netdelay_list))

    plt.plot(num_r2, test_r2_netdelay_list, label='test_r2_netdelay')
    plt.plot(num_r2, test_r2_celldelay_list, label='test_r2_celldelay')
    plt.plot(num_r2, test_r2_slew_list, label='test_r2_slew_list')
    plt.plot(num_r2, train_r2_netdelay_list, label='train_r2_netdelay')
    plt.plot(num_r2, train_r2_celldelay_list, label='train_r2_celldelay')
    plt.plot(num_r2, train_r2_slew_list, label='train_r2_slew_list')

    plt.title('R2 history')
    plt.legend()
    plt.xlabel('time/200epoch')
    plt.ylabel('r2score')
    plt.ylim(0.5, 1)
    plt.xlim(5, num_r2[-1] + 1)
    if isSave:
        plt.savefig('./data/pic/{}_r2_net_celldelay_slew.png'.format(checkpoint))
    plt.show()

    plt.plot(num_r2, test_r2_at_list, label='test_r2_at_list')
    plt.plot(num_r2, test_r2_rat_list, label='test_r2_rat_list')
    plt.plot(num_r2, train_r2_at_list, label='train_r2_at_list')
    plt.plot(num_r2, train_r2_rat_list, label='train_r2_rat_list')
    plt.title('R2 history')
    plt.legend()
    plt.xlabel('time/200epoch')
    plt.ylabel('r2score')
    plt.ylim(0.5, 1)
    plt.xlim(5, num_r2[-1] + 1)
    if isSave:
        plt.savefig('./data/pic/{}_r2_at_rat.png'.format(checkpoint))
    plt.show()

    if isSave:
        np.savetxt('./data/datalist/{}_test_r2_netdelay_list.txt'.format(checkpoint), test_r2_netdelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_test_r2_celldelay_list.txt'.format(checkpoint), test_r2_celldelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_test_r2_at_list.txt'.format(checkpoint), test_r2_at_list, delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_test_r2_slew_list.txt'.format(checkpoint), test_r2_slew_list, delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_test_r2_rat_list.txt'.format(checkpoint), test_r2_rat_list, delimiter=' ',
                   fmt='%f')

        np.savetxt('./data/datalist/{}_train_r2_netdelay_list.txt'.format(checkpoint), train_r2_netdelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_train_r2_celldelay_list.txt'.format(checkpoint), train_r2_celldelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_train_r2_at_list.txt'.format(checkpoint), train_r2_at_list, delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_train_r2_slew_list.txt'.format(checkpoint), train_r2_slew_list, delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_train_r2_rat_list.txt'.format(checkpoint), train_r2_rat_list, delimiter=' ',
                   fmt='%f')


def report_loss_withslack(checkpoint, isSave=False):
    print("train_loss_netdelay_list", train_loss_netdelay_list)
    print("test_loss_netdelay_list", eval_loss_netdelay_list)
    print("train_loss_celldelay_list", train_loss_celldelay_list)
    print("test_loss_celldelay_list", eval_loss_celldelay_list)
    print("train_loss_at_list", train_loss_at_list)
    print("test_loss_at_list", eval_loss_at_list)
    print("train_loss_slew_list", train_loss_slew_list)
    print("test_loss_slew_list", eval_loss_slew_list)
    print("train_loss_rat_list", train_loss_rat_list)
    print("test_loss_rat_list", eval_loss_rat_list)
    print("train_loss_slack_list", train_loss_slack_list)
    print("test_loss_slack_list", eval_loss_slack_list)

    num_loss_list = range(0, len(train_loss_netdelay_list))
    plt.plot(num_loss_list, train_loss_netdelay_list, label='train_loss_netdelay_list')
    plt.plot(num_loss_list, train_loss_celldelay_list, label='train_loss_celldelay_list')
    plt.plot(num_loss_list, train_loss_slew_list, label='train_loss_slew_list')

    plt.title('loss_netdelay_celldelay_slew')
    plt.legend()
    plt.xlabel('time/50epoch')
    plt.ylabel('mse loss')
    plt.ylim(0, 0.5)
    plt.xlim(0, num_loss_list[-1] + 1)
    if isSave:
        plt.savefig('./data/pic/{}_loss_netdelay_celldelay_slew.png'.format(checkpoint))
    plt.show()

    plt.plot(num_loss_list, train_loss_at_list, label='train_loss_at_list')
    plt.plot(num_loss_list, train_loss_rat_list, label='train_loss_rat_list')
    plt.plot(num_loss_list, train_loss_slack_list, label='train_loss_slack_list')
    plt.title('loss_at_rat_slack')
    plt.legend()
    plt.xlabel('time/50epoch')
    plt.ylabel('mse loss')
    plt.ylim(0, 0.5)
    plt.xlim(5, num_loss_list[-1] + 1)
    if isSave:
        plt.savefig('./data/pic/{}_loss_at_rat.png'.format(checkpoint))
    plt.show()

    if isSave:
        np.savetxt('./data/datalist/{}_train_loss_netdelay_list.txt'.format(checkpoint), train_loss_netdelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_test_loss_netdelay_list.txt'.format(checkpoint), eval_loss_netdelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_train_loss_celldelay_list.txt'.format(checkpoint), train_loss_celldelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_test_loss_celldelay_list.txt'.format(checkpoint), eval_loss_celldelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_train_loss_at_list.txt'.format(checkpoint), train_loss_at_list, delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_test_loss_at_list.txt'.format(checkpoint), eval_loss_at_list, delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_train_loss_slew_list.txt'.format(checkpoint), train_loss_slew_list,
                   delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_test_loss_slew_list.txt'.format(checkpoint), eval_loss_slew_list, delimiter=' ',
                   fmt='%f')

        np.savetxt('./data/datalist/{}_train_loss_rat_list.txt'.format(checkpoint), train_loss_rat_list,
                   delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_test_loss_rat_list.txt'.format(checkpoint), eval_loss_rat_list, delimiter=' ',
                   fmt='%f')

        np.savetxt('./data/datalist/{}_train_loss_slack_list.txt'.format(checkpoint), train_loss_slack_list,
                   delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_test_loss_slack_list.txt'.format(checkpoint), eval_loss_slack_list,
                   delimiter=' ',
                   fmt='%f')


def report_r2_withslack(checkpoint, isSave=False):
    print("test_r2_netdelay_list", test_r2_netdelay_list)
    print("test_r2_celldelay_list", test_r2_celldelay_list)
    print("test_r2_at_list", test_r2_at_list)
    print("test_r2_slew_list", test_r2_slew_list)
    print("test_r2_rat_list", test_r2_rat_list)
    print("test_r2_slack_f_list", test_r2_slack_f_list)
    print("test_r2_slack_unf_list", test_r2_slack_unf_list)
    print("test_r2_slack_ep_f_list", test_r2_slack_ep_f_list)
    print("test_r2_slack_ep_unf_list", test_r2_slack_ep_unf_list)

    # print("train_r2_netdelay_list", train_r2_netdelay_list)
    # print("train_r2_celldelay_list", train_r2_celldelay_list)
    # print("train_r2_at_list", train_r2_at_list)
    # print("train_r2_slew_list", train_r2_slew_list)
    # print("train_r2_rat_list", train_r2_rat_list)

    num_r2 = range(0, len(train_r2_netdelay_list))

    plt.plot(num_r2, test_r2_netdelay_list, label='test_r2_netdelay')
    plt.plot(num_r2, test_r2_celldelay_list, label='test_r2_celldelay')
    plt.plot(num_r2, test_r2_slew_list, label='test_r2_slew_list')
    plt.plot(num_r2, train_r2_netdelay_list, label='train_r2_netdelay')
    plt.plot(num_r2, train_r2_celldelay_list, label='train_r2_celldelay')
    plt.plot(num_r2, train_r2_slew_list, label='train_r2_slew_list')

    plt.title('R2 history')
    plt.legend()
    plt.xlabel('time/200epoch')
    plt.ylabel('r2score')
    plt.ylim(0.5, 1)
    plt.xlim(5, num_r2[-1] + 1)
    if isSave:
        plt.savefig('./data/pic/{}_r2_net_celldelay_slew.png'.format(checkpoint))
    plt.show()

    plt.plot(num_r2, test_r2_at_list, label='test_r2_at_list')
    plt.plot(num_r2, test_r2_rat_list, label='test_r2_rat_list')
    plt.plot(num_r2, train_r2_at_list, label='train_r2_at_list')
    plt.plot(num_r2, train_r2_rat_list, label='train_r2_rat_list')
    plt.title('R2 history')
    plt.legend()
    plt.xlabel('time/200epoch')
    plt.ylabel('r2score')
    plt.ylim(0.5, 1)
    plt.xlim(5, num_r2[-1] + 1)
    if isSave:
        plt.savefig('./data/pic/{}_r2_at_rat.png'.format(checkpoint))
    plt.show()

    plt.plot(num_r2, test_r2_slack_f_list, label='test_r2_slack_f_list')
    plt.plot(num_r2, test_r2_slack_ep_f_list, label='test_r2_slack_ep_f_list')
    plt.plot(num_r2, train_r2_slack_f_list, label='train_r2_slack_f_list')
    plt.plot(num_r2, train_r2_slack_ep_f_list, label='train_r2_slack_ep_f_list')
    plt.title('R2 history')
    plt.legend()
    plt.xlabel('time/200epoch')
    plt.ylabel('r2score')
    plt.ylim(0.5, 1)
    plt.xlim(5, num_r2[-1] + 1)
    if isSave:
        plt.savefig('./data/pic/{}_r2_slack_flatten.png'.format(checkpoint))
    plt.show()

    plt.plot(num_r2, test_r2_slack_unf_list, label='test_r2_slack_unf_list')
    plt.plot(num_r2, test_r2_slack_ep_unf_list, label='test_r2_slack_ep_unf_list')
    plt.plot(num_r2, train_r2_slack_unf_list, label='train_r2_slack_unf_list')
    plt.plot(num_r2, train_r2_slack_ep_unf_list, label='train_r2_slack_ep_unf_list')
    plt.title('R2 history')
    plt.legend()
    plt.xlabel('time/200epoch')
    plt.ylabel('r2score')
    plt.ylim(0.5, 1)
    plt.xlim(5, num_r2[-1] + 1)
    if isSave:
        plt.savefig('./data/pic/{}_r2_slack_unflatten.png'.format(checkpoint))
    plt.show()

    if isSave:
        np.savetxt('./data/datalist/{}_test_r2_netdelay_list.txt'.format(checkpoint), test_r2_netdelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_test_r2_celldelay_list.txt'.format(checkpoint), test_r2_celldelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_test_r2_at_list.txt'.format(checkpoint), test_r2_at_list, delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_test_r2_slew_list.txt'.format(checkpoint), test_r2_slew_list, delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_test_r2_rat_list.txt'.format(checkpoint), test_r2_rat_list, delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_test_r2_slack_f_list.txt'.format(checkpoint), test_r2_slack_f_list,
                   delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_test_r2_slack_unf_list.txt'.format(checkpoint), test_r2_slack_unf_list,
                   delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_test_r2_slack_ep_f_list.txt'.format(checkpoint), test_r2_slack_ep_f_list,
                   delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_test_r2_slack_ep_unf_list.txt'.format(checkpoint), test_r2_slack_ep_unf_list,
                   delimiter=' ',
                   fmt='%f')

        ###################################################################
        np.savetxt('./data/datalist/{}_train_r2_netdelay_list.txt'.format(checkpoint), train_r2_netdelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_train_r2_celldelay_list.txt'.format(checkpoint), train_r2_celldelay_list,
                   delimiter=' ', fmt='%f')
        np.savetxt('./data/datalist/{}_train_r2_at_list.txt'.format(checkpoint), train_r2_at_list, delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_train_r2_slew_list.txt'.format(checkpoint), train_r2_slew_list, delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_train_r2_rat_list.txt'.format(checkpoint), train_r2_rat_list, delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_train_r2_slack_f_list.txt'.format(checkpoint), train_r2_slack_f_list,
                   delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_train_r2_slack_unf_list.txt'.format(checkpoint), train_r2_slack_unf_list,
                   delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_train_r2_slack_ep_f_list.txt'.format(checkpoint), train_r2_slack_ep_f_list,
                   delimiter=' ',
                   fmt='%f')
        np.savetxt('./data/datalist/{}_train_r2_slack_ep_unf_list.txt'.format(checkpoint), train_r2_slack_ep_unf_list,
                   delimiter=' ',
                   fmt='%f')


def getData(checkpoints, isgetloss=True, isgetSigleSlack=True, isRat=False):
    with open(r'../data/datalist/{}_test_r2_netdelay_list.txt'.format(checkpoints), 'r') as fp:
        for line in fp:
            x = line[:-1]
            test_r2_netdelay_list.append(float(x))

    with open(r'../data/datalist/{}_test_r2_celldelay_list.txt'.format(checkpoints), 'r') as fp:
        for line in fp:
            x = line[:-1]
            test_r2_celldelay_list.append(float(x))

    with open(r'../data/datalist/{}_test_r2_at_list.txt'.format(checkpoints), 'r') as fp:
        for line in fp:
            x = line[:-1]
            test_r2_at_list.append(float(x))

    with open(r'../data/datalist/{}_test_r2_slew_list.txt'.format(checkpoints), 'r') as fp:
        for line in fp:
            x = line[:-1]
            test_r2_slew_list.append(float(x))

    with open(r'../data/datalist/{}_train_r2_netdelay_list.txt'.format(checkpoints), 'r') as fp:
        for line in fp:
            x = line[:-1]
            train_r2_netdelay_list.append(float(x))

    with open(r'../data/datalist/{}_train_r2_celldelay_list.txt'.format(checkpoints), 'r') as fp:
        for line in fp:
            x = line[:-1]
            train_r2_celldelay_list.append(float(x))

    with open(r'../data/datalist/{}_train_r2_at_list.txt'.format(checkpoints), 'r') as fp:
        for line in fp:
            x = line[:-1]
            train_r2_at_list.append(float(x))

    with open(r'../data/datalist/{}_train_r2_slew_list.txt'.format(checkpoints), 'r') as fp:
        for line in fp:
            x = line[:-1]
            train_r2_slew_list.append(float(x))

    if isRat:
        with open(r'../data/datalist/{}_train_r2_rat_list.txt'.format(checkpoints), 'r') as fp:
            for line in fp:
                x = line[:-1]
                train_r2_rat_list.append(float(x))

        with open(r'../data/datalist/{}_test_r2_rat_list.txt'.format(checkpoints), 'r') as fp:
            for line in fp:
                x = line[:-1]
                test_r2_rat_list.append(float(x))

    if isgetSigleSlack:
        with open(r'../data/datalist/{}_test_r2_slack_list.txt'.format(checkpoints), 'r') as fp:
            for line in fp:
                x = line[:-1]
                test_r2_slack_list.append(float(x))
        with open(r'../data/datalist/{}_train_r2_slack_list.txt'.format(checkpoints), 'r') as fp:
            for line in fp:
                x = line[:-1]
                train_r2_slack_list.append(float(x))
    else:
        with open(r'../data/datalist/{}_test_r2_slack_f_list.txt'.format(checkpoints), 'r') as fp:
            for line in fp:
                x = line[:-1]
                test_r2_slack_f_list.append(float(x))
        with open(r'../data/datalist/{}_train_r2_slack_f_list.txt'.format(checkpoints), 'r') as fp:
            for line in fp:
                x = line[:-1]
                train_r2_slack_f_list.append(float(x))

        with open(r'../data/datalist/{}_test_r2_slack_unf_list.txt'.format(checkpoints), 'r') as fp:
            for line in fp:
                x = line[:-1]
                test_r2_slack_unf_list.append(float(x))
        with open(r'../data/datalist/{}_train_r2_slack_unf_list.txt'.format(checkpoints), 'r') as fp:
            for line in fp:
                x = line[:-1]
                train_r2_slack_unf_list.append(float(x))

        with open(r'../data/datalist/{}_test_r2_slack_ep_f_list.txt'.format(checkpoints), 'r') as fp:
            for line in fp:
                x = line[:-1]
                test_r2_slack_ep_f_list.append(float(x))
        with open(r'../data/datalist/{}_train_r2_slack_ep_f_list.txt'.format(checkpoints), 'r') as fp:
            for line in fp:
                x = line[:-1]
                train_r2_slack_ep_f_list.append(float(x))

        with open(r'../data/datalist/{}_test_r2_slack_ep_unf_list.txt'.format(checkpoints), 'r') as fp:
            for line in fp:
                x = line[:-1]
                test_r2_slack_ep_unf_list.append(float(x))
        with open(r'../data/datalist/{}_train_r2_slack_ep_unf_list.txt'.format(checkpoints), 'r') as fp:
            for line in fp:
                x = line[:-1]
                train_r2_slack_ep_unf_list.append(float(x))

    if isgetloss:
        with open(r'../data/datalist/{}_train_loss_netdelay_list.txt'.format(checkpoints), 'r') as fp:
            for line in fp:
                x = line[:-1]
                train_loss_netdelay_list.append(float(x))

        with open(r'../data/datalist/{}_train_loss_celldelay_list.txt'.format(checkpoints), 'r') as fp:
            for line in fp:
                x = line[:-1]
                train_loss_celldelay_list.append(float(x))

        with open(r'../data/datalist/{}_train_loss_at_list.txt'.format(checkpoints), 'r') as fp:
            for line in fp:
                x = line[:-1]
                train_loss_at_list.append(float(x))

        with open(r'../data/datalist/{}_train_loss_slew_list.txt'.format(checkpoints), 'r') as fp:
            for line in fp:
                x = line[:-1]
                train_loss_slew_list.append(float(x))

        with open(r'../data/datalist/{}_test_loss_netdelay_list.txt'.format(checkpoints), 'r') as fp:
            for line in fp:
                x = line[:-1]
                eval_loss_netdelay_list.append(float(x))

        with open(r'../data/datalist/{}_test_loss_celldelay_list.txt'.format(checkpoints), 'r') as fp:
            for line in fp:
                x = line[:-1]
                eval_loss_celldelay_list.append(float(x))

        with open(r'../data/datalist/{}_test_loss_at_list.txt'.format(checkpoints), 'r') as fp:
            for line in fp:
                x = line[:-1]
                eval_loss_at_list.append(float(x))

        with open(r'../data/datalist/{}_test_loss_slew_list.txt'.format(checkpoints), 'r') as fp:
            for line in fp:
                x = line[:-1]
                eval_loss_slew_list.append(float(x))


def getBestCheckpoint(isBestSigleSlack=True, isRat=False, isSecond=False):
    test_avg_r2 = []
    num_r2 = range(0, len(train_r2_netdelay_list))
    if not isRat:
        for i in num_r2:
            test_avg_r2.append(
                (test_r2_netdelay_list[i] + test_r2_celldelay_list[i] + test_r2_at_list[i] + test_r2_slew_list[i]) / 4
            )
        index = test_avg_r2.index(max(test_avg_r2))
        print("max avg_test_r2score = {:.15f}, pth = {}".format(max(test_avg_r2), (index + 1) * 200 - 1))
        print("test: r2_netdelay = {}, r2_celldelay = {}, r2_at = {}, r2_slew = {}".format(test_r2_netdelay_list[index],
                                                                                           test_r2_celldelay_list[
                                                                                               index],
                                                                                           test_r2_at_list[index],
                                                                                           test_r2_slew_list[index]))

        print(
            "train: r2_netdelay = {}, r2_celldelay = {}, r2_at = {}, r2_slew = {}".format(train_r2_netdelay_list[index],
                                                                                          train_r2_celldelay_list[
                                                                                              index],
                                                                                          train_r2_at_list[index],
                                                                                          train_r2_slew_list[index]))

    else:
        for i in num_r2:
            test_avg_r2.append(
                (test_r2_netdelay_list[i] + test_r2_celldelay_list[i] + test_r2_at_list[i] + test_r2_slew_list[i] +
                 test_r2_rat_list[i]) / 5
            )
        index = test_avg_r2.index(max(test_avg_r2))
        print("max avg_test_r2score = {:.15f}, pth = {}".format(max(test_avg_r2), (index + 1) * 200 - 1))
        print("test: r2_netdelay = {}, r2_celldelay = {}, r2_at = {}, r2_slew = {}, r2_rat = {}".format(
            test_r2_netdelay_list[index],
            test_r2_celldelay_list[index],
            test_r2_at_list[index],
            test_r2_slew_list[index],
            test_r2_rat_list[index]))

        print("train: r2_netdelay = {}, r2_celldelay = {}, r2_at = {}, r2_slew = {}, r2_rat = {}".format(
            train_r2_netdelay_list[index],
            train_r2_celldelay_list[index],
            train_r2_at_list[index],
            train_r2_slew_list[index],
            train_r2_rat_list[index]))

    print("-------------------------------------------------------------------------------------------------")
    index = test_r2_at_list.index(max(test_r2_at_list))
    print("max test_r2_at_r2score = {:.15f}, pth = {}".format(test_r2_at_list[index], (index + 1) * 200 - 1))

    sort_list_at = sorted(test_r2_at_list)
    second_at_value = sort_list_at[-2]
    second_at_index = test_r2_at_list.index(second_at_value)

    print("second test_r2_at_r2score = {:.15f}, pth = {}".format(test_r2_at_list[second_at_index],
                                                                 (second_at_index + 1) * 200 - 1))

    print("-------------------------------------------------------------------------------------------------")
    if isBestSigleSlack:
        index = test_r2_slack_list.index(max(test_r2_slack_list))
        print("max test_slack_r2score = {:.15f}, pth = {}".format(max(test_r2_slack_list), (index + 1) * 200 - 1))
        print("test: r2_at = {}, r2_slack = {}".format(test_r2_at_list[index], test_r2_slack_list[index]))
        print("train: r2_at = {}, r2_slack = {}".format(train_r2_at_list[index], train_r2_slack_list[index]))

    else:
        index = test_r2_slack_f_list.index(max(test_r2_slack_f_list))
        print("max r2_score test_r2_slack_f_list = {:.15f}, pth = {}".format(max(test_r2_slack_f_list),
                                                                             (index + 1) * 200 - 1))
        index = test_r2_slack_unf_list.index(max(test_r2_slack_unf_list))
        print("max r2_score test_r2_slack_unf_list = {:.15f}, pth = {}".format(max(test_r2_slack_unf_list),
                                                                               (index + 1) * 200 - 1))
        index = test_r2_slack_ep_f_list.index(max(test_r2_slack_ep_f_list))
        print("max r2_score test_r2_slack_ep_f_list = {:.15f}, pth = {}".format(max(test_r2_slack_ep_f_list),
                                                                                (index + 1) * 200 - 1))
        index = test_r2_slack_ep_unf_list.index(max(test_r2_slack_ep_unf_list))
        print("max r2_score test_r2_slack_ep_unf_list = {:.15f}, pth = {}".format(max(test_r2_slack_ep_unf_list),
                                                                                  (index + 1) * 200 - 1))
        print("-------------------------------------------------------------------------------------------------")
        test_avg_atslackef = []
        for i in num_r2:
            test_avg_atslackef.append(
                (test_r2_at_list[i] + test_r2_slack_ep_f_list[i]) / 2
            )
        index = test_avg_atslackef.index(max(test_avg_atslackef))
        print("max test_avg_atslackef = {:.15f}, pth = {}".format(max(test_avg_atslackef), (index + 1) * 200 - 1))
    if isRat:
        print("-------------------------------------------------------------------------------------------------")
        index = test_r2_rat_list.index(max(test_r2_rat_list))
        print("max r2_score test_r2_rat_list = {:.15f}, pth = {}".format(max(test_r2_rat_list),
                                                                         (index + 1) * 200 - 1))

    if isSecond:
        sort_avg = sorted(test_avg_r2)
        second_value = sort_avg[-2]
        second_index = test_avg_r2.index(second_value)

        print("-------------------------------------------------------------------------------------------------")
        print("second avg_test_r2score = {:.15f}, pth = {}".format(second_value, (second_index + 1) * 200 - 1))
        print("test: r2_netdelay = {}, r2_celldelay = {}, r2_at = {}, r2_slew = {}".format(
            test_r2_netdelay_list[second_index],
            test_r2_celldelay_list[
                second_index],
            test_r2_at_list[second_index],
            test_r2_slew_list[second_index]))


# 0507_PreRoutSGAT3_lr3_e12000_rd11_bm0_load2_4 #5399
# 0507_PreRoutSGAT3_lr3_e12000_rd11_bm0_load_2 #1599
# 0506_PreRoutSGAT3_lr3_e12000_rd11_bm0_load0_1 # 1399
# 0505_PreRoutSGAT3_lr10_e12000_rd7_bm0_load1_2 #9599,3199
# 0428_PreRoutSGAT3_lr3_e10000_rdgly_gt1_bm0_load0_2 pth799.at=0.914,avg=88 pth1399
# 0429_PreRoutSGAT3_lr3_e10000_rd42_bm0_load2rdgly_3 pth8799.at=0.907, avg = 0.88, pth1199.at=0.917


# 0507_PreRoutSGAT4_lr3_e12000_rd13_bm0_load1_2 pth=10999,11999
# 0506_PreRoutSGAT1_lr10_e12000_rd11_bm0_load0_1 pth=9999,10399,9399#model name false
# 0504_PreRoutSGAT4_lr3_e15000_rd7_bm0_load0_1 pth=13399,13199
# 0429_PreRoutSGAT4_lr3_e10000_rd36_bm0_load0_1 pth9599.at=93.2 avg=93.1  pth7799
# 0428_PreRoutSGAT4_lr10_e10000_rd42_bm0_load0_1 pth9999.at=91.5
# 0428_PreRoutSGAT4_lr3_e10000_rdgly_bm0_load2_3 pth6399.at=92.9 avg=92.7 pth8799

# 0507_PreRoutSGAT1_lr10_e12000_rd17_bm0_load1_0,pth:10399,11999
# 0507_PreRoutSGAT1_lr3_e20000_rd13_bm0_load0_1,pth:13399,15599,17599
# 0506_PreRoutSGAT1_lr3_e45000_rd11_bm0,pth34799,34999
# 0427_PreRoutSGAT1_lr4_e4000_rdgly_bm0_load3_5 pth3999.at=0.884,avg=0.92
# 0428_PreRoutSGAT1_lr3_e10000_rd42_bm0_load1_2 pth9199.at=0.89, avg=0.92
# 0429_PreRoutSGAT1_lr10_e10000_rd36_bm0_load0_1 pth9399.at=0.90, avg =0.923
# 0503_PreRoutSGAT1_lr3_e15000_rd7_bm0_load0_2
if __name__ == '__main__':
    checkpoints = "0506_PreRoutSGAT3_lr3_e12000_rd11_bm0_load0_1"
    checkpoints += '_loopgt0'  # _loopgt0, _loopgt0_new

    getData(checkpoints, isgetloss=False, isgetSigleSlack=False, isRat=False)  # True)  # False)  # True)  # False)
    getBestCheckpoint(isBestSigleSlack=False, isRat=False, isSecond=True)  # False)  # True)  # True)  # False)

    # report_r2(checkpoints, isSave=False, isRat=True)
    # report_slack(checkpoints, isSave=False, isSingle=False)
