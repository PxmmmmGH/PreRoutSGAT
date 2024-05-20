import torch
import dgl
import random
import time
import numpy as np

import sys

sys.path.append("..")

no_log = False  # True


def fix_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


fix_benchmark = 0  # False  # True
# fix_random(8026728)
# fix_random(103108121)
# fix_random(42)
# fix_random(42)
# fix_random(36)
# fix_random(7)
# fix_random(11)
# fix_random(13)
fix_random(17)

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


# <editor-fold desc="data processing">
data = {}
for k in available_data:
    g = dgl.load_graphs('./data/8_rat/{}.graph.bin'.format(k))[0][0].to('cuda')

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

    # add is 'is timing endpoint' to nf
    # is_timing_endpt = g.ndata['n_is_timing_endpt'].reshape(-1, 1)
    # g.ndata['nf'] = torch.cat([g.ndata['nf'], is_timing_endpt], dim=1)

    ## add example arc
    # arc_list = []
    # net_src_nodes, net_dst_nodes = g.edges(etype='net_out', #form='uv')
    # cell_src_nodes, cell_dst_nodes = g.edges(etype='cell_out', #form='uv')
    # net_src_nodes = list(np.array(net_src_nodes.cpu()))
    # net_dst_nodes = list(np.array(net_dst_nodes.cpu()))
    # cell_src_nodes = list(np.array(cell_src_nodes.cpu()))
    # cell_dst_nodes = list(np.array(cell_dst_nodes.cpu()))
    # node = ts['topo'][0][0]
    # arc_list.append(node)
    # for i in range(1, len(ts['topo'])):
    #    print(i)
    #    if i % 2 == 1:
    #        if node not in net_src_nodes:
    #            break
    #        index = net_src_nodes.index(node)
    #        node = net_dst_nodes[index]
    #        arc_list.append(node)
    #    else:
    #        if node not in cell_src_nodes:
    #            break
    #        index = cell_src_nodes.index(node)
    #        node = cell_dst_nodes[index]
    #        arc_list.append(node)

    g.edges['cell_out'].data['ef'] = g.edges['cell_out'].data['ef'].type(torch.float32)
    g.edges['cell_out'].data['e_cell_delays'] = g.edges['cell_out'].data['e_cell_delays'].type(torch.float32)
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
          'topo_time': topo_time,
          # 'example arc': arc_list
          }
    data[k] = g, ts

data_train = {k: t for k, t in data.items() if k in train_data_keys}
data_test = {k: t for k, t in data.items() if k not in train_data_keys}

# </editor-fold>
output_node_netdelay_log = torch.log(0.0001 + torch.zeros(1, 1)) + 7.6

if __name__ == '__main__':
    # print('Graph statistics: (total {} graphs)'.format(len(data)))
    # print('{:15} {:>10} {:>10}'.format('NAME', '#NODES', '#EDGES'))
    # for k, (g, ts) in data.items():
    #     print('{:15} {:>10} {:>10}'.format(k, g.num_nodes(), g.num_edges()))

    for dic in [data_train, data_test]:
        for k, (g, ts) in dic.items():
            ...
            # print("-----------------", k, "----------------")
            # print("num_node=", g.num_nodes())
            # print("level=", len(ts["topo"]))
            # print("endpt_num = ", torch.sum(g.ndata['n_is_timing_endpt'] == True).item())
    #         print('\\texttt{{{}}},{},{},{},{},{},{}'.format(k.replace('_', '\_'), g.num_nodes(), g.num_edges('net_out'), g.num_edges('cell_out'), len(ts['topo']), len(ts['po_nodes']), len(ts['endpoints'])))

    g, ts = data_train["blabla"]
    pi_star_node = ts['topo'][0][0]
    node_drive = pi_star_node
    arc_list = []
    for i in range(1, len(ts['topo'])):
        if i % 2 == 1:
            es = g.in_edges(node_drive, etype='net_out')
            node_drive = es[1][0]
            arc_list.append(node_drive)
        else:
            es = g.in_edges(node_drive, etype='cell_out')
            node_drive = es[1][0]
            arc_list.append(node_drive)

    # print(g.edges['cell_out'].data['e_cell_delays'].shape)
    # print(torch.sum(torch.isinf(g.ndata['n_rats']) == True))
    # g.ndata['n_rats'][torch.isinf(g.ndata['n_rats'])] = 0
    # print(torch.sum(torch.isinf(g.ndata['n_rats']) == True))
    # print(ts["topo"])
    # print(g.ndata)
    # print(g.ndata['n_is_timing_endpt'].dtype)
    # test = g.ndata['n_is_timing_endpt'].bool()
    # print(test.dtype)
    # print(test)
    # b = g.ndata['nf'][g.ndata['n_is_timing_endpt'].bool()]
    # print(g.ndata['nf'][g.ndata['n_is_timing_endpt'].bool()])
    # print(torch.sum(g.ndata['n_is_timing_endpt'] == True).item(), len(b))
# n_rats
# n_net_delays
# n_ats
# n_slews
# n_slew_log
# nf
# n_is_timing_endpt
# n_net_delays_log

...
