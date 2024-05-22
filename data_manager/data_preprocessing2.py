# @Time :4/22/2024 5:28 PM
# @Author :Pxmmmm
# @File :data_preprocessing2.py
# @Version:  0.1
# @Conten: add cell in

import torch
import dgl
import random
import time
import numpy as np

import sys

sys.path.append("..")


def fix_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


no_log = False  # True
fix_benchmark = 0  # 2  # False  # True
is_Cell_IN = True  # False  # True
# fix_random(8026728)
# fix_random(103108121)
# fix_random(36)
# fix_random(42)
# fix_random(42)
# fix_random(7)
# fix_random(11)
# fix_random(13)
fix_random(17)

is_arc_path = False  # True

available_data = 'blabla usb_cdc_core BM64 jpeg_encoder salsa20 usbf_device aes128 wbqspiflash aes192 cic_decimator xtea aes256 des spm y_huff aes_cipher picorv32a synth_ram zipdiv genericfir usb'.split()

if fix_benchmark == 0:
    train_data_keys = 'blabla usb_cdc_core BM64 salsa20 aes128 wbqspiflash cic_decimator aes256 des aes_cipher picorv32a zipdiv genericfir usb'.split()
elif fix_benchmark == 1:
    train_data_keys = 'blabla usb_cdc_core BM64 jpeg_encoder salsa20 usbf_device aes128 wbqspiflash aes192 cic_decimator xtea aes256 des spm'.split()
elif fix_benchmark == 2:
    train_data_keys = 'blabla usb_cdc_core BM64 usb salsa20 usbf_device jpeg_encoder wbqspiflash aes192 cic_decimator xtea aes256 des spm'.split()

else:
    train_data_keys = random.sample(available_data, 14)


def gen_topo(g_hetero):
    torch.cuda.synchronize()
    time_s = time.time()
    na, nb = g_hetero.edges(etype='net_out', form='uv')
    ca, cb = g_hetero.edges(etype='cell_out', form='uv')
    g = dgl.graph((torch.cat([na, ca]).cpu(), torch.cat([nb, cb]).cpu()))
    topo = dgl.topological_nodes_generator(g)
    ret = [t.cuda() for t in topo]
    torch.cuda.synchronize()
    time_e = time.time()
    return ret, time_e - time_s


def gen_homograph_with_features(g_hetero):
    # for DeepGCNII baseline
    na, nb = g_hetero.edges(etype='net_out', form='uv')
    ca, cb = g_hetero.edges(etype='cell_out', form='uv')
    g = dgl.graph((torch.cat([na, ca]), torch.cat([nb, cb])))

    # node feature
    g.ndata['nf'] = g_hetero.ndata['nf']
    g.ndata['n_ats'] = g_hetero.ndata['n_ats']
    g.ndata['n_slew_log'] = g_hetero.ndata['n_slew_log']
    g.ndata['n_net_delays_log'] = g_hetero.ndata['n_net_delays_log']
    g.ndata['n_rats'] = g_hetero.ndata['n_rats']
    g.ndata['n_is_timing_endpt'] = g_hetero.ndata['n_is_timing_endpt']

    # cell feature
    is_net_edge = torch.zeros(g.num_edges(), 1).to("cuda:0")
    is_net_edge[:len(na), :] = 1
    is_cell_edge = torch.zeros(g.num_edges(), 1).to("cuda:0")
    is_cell_edge[len(na):, :] = 1
    net_ef = g_hetero.edges['net_out'].data['ef']
    cell_ef = g_hetero.edges['cell_out'].data['ef']
    ne = torch.cat([net_ef, torch.zeros(net_ef.shape[0], cell_ef.shape[1]).to("cuda:0")], dim=1)
    ce = torch.cat([torch.zeros(cell_ef.shape[0], net_ef.shape[1]).to("cuda:0"), cell_ef], dim=1)
    g.edata['ef'] = torch.cat([ne, ce], dim=0)
    g.edata['ef'] = torch.cat([g.edata['ef'], is_net_edge, is_cell_edge], dim=1)

    return g


def create_graph_with_cellin():
    g0 = dgl.load_graphs('../data/8_rat/{}.graph.bin'.format(k))[0][0].to('cuda')
    ni_src, ni_dst = g0.edges(etype='net_in', form='uv')
    no_src, no_dst = g0.edges(etype='net_out', form='uv')
    co_src, co_dst = g0.edges(etype='cell_out', form='uv')
    ci_src = co_dst
    ci_dst = co_src

    g = dgl.heterograph({
        ('node', 'net_out', 'node'): (no_src, no_dst),
        ('node', 'net_in', 'node'): (ni_src, ni_dst),
        ('node', 'cell_out', 'node'): (co_src, co_dst),
        ('node', 'cell_in', 'node'): (ci_src, ci_dst)  # 新添加的反向边
    })

    g.ndata['nf'] = g0.ndata['nf']
    g.ndata['n_rats'] = g0.ndata['n_rats']
    g.ndata['n_net_delays'] = g0.ndata['n_net_delays']
    g.ndata['n_ats'] = g0.ndata['n_ats']
    g.ndata['n_slews'] = g0.ndata['n_slews']
    g.ndata['n_is_timing_endpt'] = g0.ndata['n_is_timing_endpt']

    g.edges['net_out'].data['ef'] = g0.edges['net_out'].data['ef']
    g.edges['net_in'].data['ef'] = g0.edges['net_in'].data['ef']

    g.edges['cell_out'].data['ef'] = g0.edges['cell_out'].data['ef']
    g.edges['cell_out'].data['e_cell_delays'] = g0.edges['cell_out'].data['e_cell_delays']

    # save
    dgl.save_graphs('../data/dataset2/{}.bin'.format(k), [g])

    return g


data = {}
graphlist = []
for k in available_data:
    if not is_Cell_IN:
        g = dgl.load_graphs('./data/8_rat/{}.graph.bin'.format(k))[0][0].to('cuda')
    else:
        g = dgl.load_graphs('./data/dataset2/{}.bin'.format(k))[0][0].to('cuda')

    # g = create_graph_with_cellin()

    if no_log:
        g.ndata['n_net_delays_log'] = g.ndata['n_net_delays']
    else:
        g.ndata['n_net_delays_log'] = torch.log(0.0001 + g.ndata['n_net_delays']) + 7.6
    invalid_nodes = torch.abs(g.ndata['n_ats']) > 1e20  # ignore all uninitialized stray pins
    g.ndata['n_ats'][invalid_nodes] = 0
    g.ndata['n_slews'][invalid_nodes] = 0

    # ------------------------------rat----------------------------
    # g.ndata['n_rats'][invalid_nodes] = 0
    g.ndata['n_rats'][torch.isinf(g.ndata['n_rats'])] = 0

    # expand to log
    if no_log:
        g.ndata['n_slew_log'] = g.ndata['n_slews']
    else:
        g.ndata['n_slew_log'] = torch.log(0.0001 + g.ndata['n_slews']) + 3

    g.edges['cell_out'].data['ef'] = g.edges['cell_out'].data['ef'].type(torch.float32)
    g.edges['cell_out'].data['e_cell_delays'] = g.edges['cell_out'].data['e_cell_delays'].type(torch.float32)

    # ts = np.load('../data/dataset2/{}_ts.npy'.format(k), allow_pickle=True).item()
    topo, topo_time = gen_topo(g)
    ts = {'input_nodes': (g.ndata['nf'][:, 1] < 0.5).nonzero().flatten().type(torch.int32),
          'output_nodes': (g.ndata['nf'][:, 1] > 0.5).nonzero().flatten().type(torch.int32),
          'output_nodes_nonpi': torch.logical_and(g.ndata['nf'][:, 1] > 0.5,
                                                  g.ndata['nf'][:, 0] < 0.5).nonzero().flatten().type(torch.int32),
          'pi_nodes': torch.logical_and(g.ndata['nf'][:, 1] > 0.5, g.ndata['nf'][:, 0] > 0.5).nonzero().flatten().type(
              torch.int32),
          'po_nodes': torch.logical_and(g.ndata['nf'][:, 1] < 0.5, g.ndata['nf'][:, 0] > 0.5).nonzero().flatten().type(
              torch.int32),
          'endpoints': (g.ndata['n_is_timing_endpt'] > 0.5).nonzero().flatten().type(torch.long),
          'topo': topo,
          'topo_time': topo_time
          }

    # add example arc
    if is_arc_path:
        arc_list = []
        net_src_nodes, net_dst_nodes = g.edges(etype='net_out', form='uv')
        cell_src_nodes, cell_dst_nodes = g.edges(etype='cell_out', form='uv')
        net_src_nodes = list(np.array(net_src_nodes.cpu()))
        net_dst_nodes = list(np.array(net_dst_nodes.cpu()))
        cell_src_nodes = list(np.array(cell_src_nodes.cpu()))
        cell_dst_nodes = list(np.array(cell_dst_nodes.cpu()))
        node = ts['topo'][-1][0]
        arc_list.insert(0, node.int().cpu())

        for i in range(1, len(ts['topo'])):
            # print(i)
            if i % 2 == 0:
                if node in cell_dst_nodes:
                    index = cell_dst_nodes.index(node)
                    node = cell_src_nodes[index]
                    arc_list.insert(0, node)
                else:
                    print("arc done: ", k)
                    break
            else:
                if node in net_dst_nodes:
                    index = net_dst_nodes.index(node)
                    node = net_src_nodes[index]
                    arc_list.insert(0, node)
                else:
                    print("arc done: ", k)
                    break

        ts['eg_arc'] = arc_list

        # # save
        # np.save('../data/dataset2/{}_ts.npy'.format(k), dict)

    data[k] = g, ts

data_train = {k: t for k, t in data.items() if k in train_data_keys}
data_test = {k: t for k, t in data.items() if k not in train_data_keys}

# </editor-fold>
output_node_netdelay_log = torch.log(0.0001 + torch.zeros(1, 1)) + 7.6

if __name__ == '__main__':
    for dic in [data_train, data_test]:
        for k, (g, ts) in dic.items():
            print(ts['eg_arc'])
            # print(k, g.edges['net_out'].data.keys())
            # print(k, g.edges['cell_out'].data.keys())
            # print(g)

# n_rats
# n_net_delays
# n_ats
# n_slews
# n_slew_log
# nf
# n_is_timing_endpt
# n_net_delays_log
