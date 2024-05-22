# @Time :4/7/2024 4:23 PM
# @Author :Pxmmmm
# @Site :
# @File :data_preprocessing_homo.py
# @Version:  0.1
import random

import torch


def gen_homograph():
    from data_preprocessing import data_train, data_test, gen_homograph_with_features
    # replace hetero graph with homographs
    # do not execute this in other modules, as it would modify
    # the global data in a dirty way
    for dic in [data_train, data_test]:
        for k in dic:
            g, ts = dic[k]
            dic[k] = gen_homograph_with_features(g)

    torch.save([data_train, data_test], '../data/7_homotest/train_test.pt')


data_train, data_test = torch.load('./data/7_homotest/train_test.pt')

if __name__ == '__main__':
    gen_homograph()

    # data_train, data_test = torch.load('../data/7_homotest/train_test.pt')
    # for k, v in random.sample(data_test.items(), 3):
    #    print(k, v)
