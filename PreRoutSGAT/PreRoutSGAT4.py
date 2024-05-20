# @Time :4/27/2024 3:56 PM
# @Author :Pxmmmm
# @File :PreRoutSGAT4.py
# @Version:  0.1
# @Conten: 3 netConv, auxiliary net delay,

import torch
import torch.nn.functional as F
import dgl
import dgl.function as fn
import functools
import pdb

import torch.nn as nn


class MLP(torch.nn.Module):
    def __init__(self, *sizes, batchnorm=False, dropout=False):
        super().__init__()
        fcs = []
        for i in range(1, len(sizes)):
            fcs.append(torch.nn.Linear(sizes[i - 1], sizes[i]))
            if i < len(sizes) - 1:
                fcs.append(torch.nn.LeakyReLU(negative_slope=0.2))
                if dropout: fcs.append(torch.nn.Dropout(p=0.2))
                if batchnorm: fcs.append(torch.nn.BatchNorm1d(sizes[i]))
        self.layers = torch.nn.Sequential(*fcs)

    def forward(self, x):
        return self.layers(x)


class NetConv(torch.nn.Module):
    def __init__(self, in_nf, in_ef, out_nf):
        super().__init__()
        self.in_nf = in_nf
        self.out_nf = out_nf
        self.in_ef = in_ef

        self.MLP_msg_net_out = MLP(self.in_nf * 2 + self.in_ef, 64, 64, 64, self.in_nf)
        self.MLP_msg_net_out_fc = nn.Linear(self.in_nf * 2, self.out_nf)
        self.MLP_reduce_net_out = MLP(self.in_nf + self.out_nf, 64, 64, 64, self.out_nf)

        self.MLP_msg_net_in = MLP(self.in_nf * 2 + self.in_ef, 64, 64, 64, self.in_nf * 2 + 1)
        self.MLP_msg_net_in_fc1 = nn.Linear(self.in_nf * 2, self.out_nf)
        self.MLP_msg_net_in_fc2 = nn.Linear(self.in_nf * 2, self.out_nf)
        self.MLP_reduce_net_in = MLP(self.in_nf + self.out_nf * 2, 64, 64, 64, self.out_nf)

    def edge_msg_net_out(self, edges):
        x = torch.cat([edges.src['nf'], edges.dst['nf'], edges.data['ef']], dim=1)
        x = self.MLP_msg_net_out(x)
        x = torch.cat([x, edges.src['nf']], dim=1)
        x = self.MLP_msg_net_out_fc(x)
        return {'efi': x}

    def node_reduce_net_out(self, nodes):
        x = torch.cat([nodes.data['nf'], nodes.data['nfi']], dim=1)
        x = self.MLP_reduce_net_out(x)
        return {'new_nf': x}

    def edge_msg_net_in(self, edges):
        x = torch.cat([edges.src['nf'], edges.dst['nf'], edges.data['ef']], dim=1)
        x = self.MLP_msg_net_in(x)
        k, f1, f2 = torch.split(x, [1, self.in_nf, self.in_nf], dim=1)
        k = torch.sigmoid(k)
        x1 = self.MLP_msg_net_in_fc1(torch.cat([f1 * k, edges.src['nf']], dim=1))
        x2 = self.MLP_msg_net_in_fc2(torch.cat([f2 * k, edges.src['nf']], dim=1))
        return {'efo1': x1, 'efo2': x2}

    def node_reduce_net_in(self, nodes):
        x = torch.cat([nodes.data['nf'], nodes.data['nfo1'], nodes.data['nfo2']], dim=1)
        x = self.MLP_reduce_net_in(x)
        return {'new_nf': x}

    def forward(self, g, ts, nf):
        with g.local_scope():
            g.ndata['nf'] = nf
            # net out
            g.apply_edges(self.edge_msg_net_out, etype='net_out')
            g.update_all(fn.copy_e('efi', 'efi'), fn.sum('efi', 'nfi'), etype='net_out')
            g.apply_nodes(self.node_reduce_net_out, ts['input_nodes'])

            # net_in
            g.apply_edges(self.edge_msg_net_in, etype='net_in')
            g.update_all(fn.copy_e('efo1', 'efo1'), fn.mean('efo1', 'nfo1'), etype='net_in')
            g.update_all(fn.copy_e('efo2', 'efo2'), fn.max('efo2', 'nfo2'), etype='net_in')
            g.apply_nodes(self.node_reduce_net_in, ts['output_nodes'])

            return g.ndata['new_nf']


class CellConv(torch.nn.Module):
    def __init__(self, in_nf, out_nf):
        super().__init__()
        self.in_nf = in_nf
        self.out_nf = out_nf

        self.MLP_msg_cell_in = MLP(self.in_nf * 2, 64, 64, 64, self.in_nf)
        self.MLP_msg_cell_in_fc = nn.Linear(self.in_nf, self.out_nf)
        self.MLP_reduce_cell_in = MLP(self.in_nf + self.out_nf, 64, 64, 64, self.out_nf)

        self.MLP_msg_cell_out = MLP(self.in_nf * 2, 64, 64, 64, self.in_nf * 2 + 1)
        self.MLP_msg_cell_out_fc1 = nn.Linear(self.in_nf, self.out_nf)
        self.MLP_msg_cell_out_fc2 = nn.Linear(self.in_nf, self.out_nf)
        self.MLP_reduce_cell_out = MLP(self.in_nf + self.out_nf * 2, 64, 64, 64, self.out_nf)

    def edge_msg_cell_in(self, edges):
        x = torch.cat([edges.src['nf'], edges.dst['nf']], dim=1)
        x = self.MLP_msg_cell_in(x)
        x = x + edges.src['nf']
        x = self.MLP_msg_cell_in_fc(x)
        return {'efi': x}

    def node_reduce_cell_in(self, nodes):
        x = torch.cat([nodes.data['nf'], nodes.data['nfi']], dim=1)
        x = self.MLP_reduce_cell_in(x)
        return {'new_nf': x}

    def edge_msg_cell_out(self, edges):
        x = torch.cat([edges.src['nf'], edges.dst['nf']], dim=1)
        x = self.MLP_msg_cell_out(x)
        k, f1, f2 = torch.split(x, [1, self.in_nf, self.in_nf], dim=1)
        k = torch.sigmoid(k)
        x1 = self.MLP_msg_cell_out_fc1(f1 * k + edges.src['nf'])
        x2 = self.MLP_msg_cell_out_fc2(f2 * k + edges.src['nf'])
        return {'efo1': x1, 'efo2': x2}

    def node_reduce_cell_out(self, nodes):
        x = torch.cat([nodes.data['nf'], nodes.data['nfo1'], nodes.data['nfo2']], dim=1)
        x = self.MLP_reduce_cell_out(x)
        return {'new_nf': x}

    def forward(self, g, ts, nf):
        with g.local_scope():
            g.ndata['nf'] = nf
            # cell in
            g.apply_edges(self.edge_msg_cell_in, etype='cell_in')
            g.update_all(fn.copy_e('efi', 'efi'), fn.sum('efi', 'nfi'), etype='cell_in')
            g.apply_nodes(self.node_reduce_cell_in, ts['input_nodes'])

            # cell out
            g.apply_edges(self.edge_msg_cell_out, etype='cell_out')
            g.update_all(fn.copy_e('efo1', 'efo1'), fn.mean('efo1', 'nfo1'), etype='cell_out')
            g.update_all(fn.copy_e('efo2', 'efo2'), fn.max('efo2', 'nfo2'), etype='cell_out')
            g.apply_nodes(self.node_reduce_cell_out, ts['output_nodes'])

            return g.ndata['new_nf']


def full_attention_conv(qs, ks, vs, output_attn=False):
    # normalize input
    qs = qs / torch.norm(qs, p=2)  # [N, H, M]
    ks = ks / torch.norm(ks, p=2)  # [L, H, M]
    N = qs.shape[0]

    # numerator
    kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
    attention_num += N * vs

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(
        attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer  # [N, H, D]

    # compute attention for visualization if needed
    if output_attn:
        attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
        normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
        attention = attention / normalizer

    if output_attn:
        return attn_output, attention
    else:
        return attn_output


class TransConvLayer(nn.Module):  # transformer with fast attention
    def __init__(self,
                 in_channels_q,
                 in_channels_k,
                 in_channels_v,
                 out_channels,
                 num_heads=1):
        super().__init__()
        self.Wq = nn.Linear(in_channels_q, out_channels * num_heads)
        self.Wk = nn.Linear(in_channels_k, out_channels * num_heads)
        self.Wv = nn.Linear(in_channels_v, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads

        self.cuda()
        self.reset_parameters()

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        self.Wv.reset_parameters()

    def forward(self, query_input, key_input, value_input):
        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(key_input).reshape(-1, self.num_heads, self.out_channels)
        value = self.Wv(value_input).reshape(-1, self.num_heads, self.out_channels)

        attention_output = full_attention_conv(query, key, value)  # [N, H, D]

        final_output = attention_output.mean(dim=1)

        return final_output


class SignalProp(torch.nn.Module):
    def __init__(self, in_nf, out_nf):
        super().__init__()
        self.in_nf = in_nf
        self.out_nf = out_nf  # 12
        self.in_cell_num_luts = 8  # 8
        self.in_cell_lut_sz = 7  # 7
        self.out_cef = 4  # 4

        self.ef_key = (1 + 2 * 7) * 8
        self.ef_value = 8 * 7 ** 2

        self.epochStep_update_topoLevel = 80
        self.num_head = 1

        # net_out prop
        self.MLP_netprop = MLP(self.out_nf + 2 * self.in_nf, 64, 64, 64, 64, self.out_nf)
        self.MLP_netprop2 = nn.Linear(2 * self.out_nf, self.out_nf)

        # cell_out prop
        self.mlp_q = MLP(self.in_nf * 2 + self.out_nf, 64, 64, 64, 64, self.out_nf)
        self.MJA = TransConvLayer(self.out_nf, self.ef_key, self.ef_value, self.out_nf, num_heads=self.num_head)
        self.layer_norm = nn.LayerNorm(self.out_nf)
        self.mlp_m = MLP(self.in_nf * 2 + self.out_nf * 2, 64, 64, 64, 64, self.out_nf + self.out_cef)
        self.mlp_a = MLP(self.in_nf + self.out_nf * 2, 64, 64, 64, 64, self.out_nf)

    def edge_msg_net(self, edges, groundtruth=False):
        if groundtruth:
            last_nf = torch.cat([edges.src['n_ats'], edges.src['n_slew_log']], dim=1)
            # last_nf = torch.cat([edges.src['n_ats'], edges.src['n_slew_log'], edges.src['n_rats']], dim=1)
        else:
            last_nf = edges.src['new_nf']

        x = torch.cat([last_nf, edges.src['nf'], edges.dst['nf']], dim=1)
        x = torch.cat([self.MLP_netprop(x), last_nf], dim=1)
        x = self.MLP_netprop2(x)
        return {'efn': x}

    def edge_msg_cell(self, edges, groundtruth=False):
        if groundtruth:
            last_nf = torch.cat([edges.src['n_ats'], edges.src['n_slew_log']], dim=1)
            # last_nf = torch.cat([edges.src['n_ats'], edges.src['n_slew_log'], edges.src['n_rats']], dim=1)
        else:
            last_nf = edges.src['new_nf']

        Qji = self.mlp_q(torch.cat([edges.src['nf'], edges.dst['nf'], last_nf], dim=1)) + last_nf
        Kji = edges.data['ef'][:, :self.ef_key]
        Vji = edges.data['ef'][:, self.ef_key:]
        Oji = self.MJA(Qji, Kji, Vji)
        Yji = self.layer_norm(Qji + Oji)
        x = self.mlp_m(torch.cat([edges.src['nf'], edges.dst['nf'], last_nf, Yji], dim=1))

        mji = x[:, :self.out_nf] + last_nf
        cell_delay = x[:, self.out_nf:]

        return {'mji': mji, 'efce': cell_delay}

    def node_reduce_o(self, nodes):
        x = self.mlp_a(torch.cat([nodes.data['nf_mean'], nodes.data['nf_max'], nodes.data['nf']], dim=1))

        return {'new_nf': x}

    def node_skip_level_o(self, nodes):
        # return {'new_nf': torch.cat([nodes.data['n_ats'], nodes.data['n_slew_log'], nodes.data['n_rats']], dim=1)}
        return {'new_nf': torch.cat([nodes.data['n_ats'], nodes.data['n_slew_log']], dim=1)}

    def forward(self, g, ts, nf, groundtruth, epoch):
        assert len(ts['topo']) % 2 == 0, 'The number of logic levels must be even (net, cell, net...)'

        with g.local_scope():
            # init level 0 with ground truth features
            g.ndata['nf'] = nf
            g.ndata['new_nf'] = torch.zeros(g.num_nodes(), self.out_nf, device='cuda', dtype=nf.dtype)
            g.apply_nodes(self.node_skip_level_o, ts['pi_nodes'])

            def prop_net(nodes, groundtruth):
                g.pull(nodes, functools.partial(self.edge_msg_net, groundtruth=groundtruth), fn.sum('efn', 'new_nf'),
                       etype='net_out')

            def prop_cell(nodes, groundtruth):
                es = g.in_edges(nodes, etype='cell_out')
                g.apply_edges(functools.partial(self.edge_msg_cell, groundtruth=groundtruth), es, etype='cell_out')
                g.send_and_recv(es, fn.copy_e('mji', 'mji'), fn.mean('mji', 'nf_mean'), etype='cell_out')
                g.send_and_recv(es, fn.copy_e('mji', 'mji'), fn.max('mji', 'nf_max'), etype='cell_out')
                g.apply_nodes(self.node_reduce_o, nodes)
                # print()

            if not groundtruth:
                # propagate
                for i in range(1, len(ts['topo'])):
                    if i < len(ts['topo']) - int(epoch / self.epochStep_update_topoLevel):
                        if i % 2 == 1:
                            prop_net(ts['topo'][i], groundtruth=True)
                        else:
                            prop_cell(ts['topo'][i], groundtruth=True)
                    else:
                        if i % 2 == 1:
                            prop_net(ts['topo'][i], groundtruth=False)
                        else:
                            prop_cell(ts['topo'][i], groundtruth=False)
            else:
                # don't need to propagate.
                prop_net(ts['input_nodes'], groundtruth)
                prop_cell(ts['output_nodes_nonpi'], groundtruth)

            return g.ndata['new_nf'], g.edges['cell_out'].data['efce']


class PreRoutSGAT4(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.nc_out = 32
        self.in_nf = 10
        self.nf_nd = 4

        self.nc1 = NetConv(self.in_nf, 2, self.nc_out)
        self.nc2 = NetConv(self.nc_out + self.in_nf, 2, self.nc_out)
        self.nc3 = NetConv(self.nc_out * 2 + self.in_nf, 2, self.nc_out)
        self.cc = CellConv(self.in_nf, self.nc_out)

        # self.nc4 = NetConv(self.in_nf + self.nf_nd, 2, 4)

        self.prop = SignalProp(self.in_nf + self.nc_out, 8)

    def forward(self, g, ts, groundtruth=False, epoch=999999):
        nf0 = g.ndata['nf']
        x1 = self.nc1(g, ts, nf0)
        x2 = self.nc2(g, ts, torch.cat([x1, nf0], dim=1))
        x3 = self.nc3(g, ts, torch.cat([x2, x1, nf0], dim=1)) + self.cc(g, ts, nf0)

        net_delays = x3[:, :self.nf_nd]  # self.nc4(g, ts, torch.cat([nf0, x3[:, :self.nf_nd]], dim=1))

        nf1 = torch.cat([nf0, x3], dim=1)
        nf2, cell_delays = self.prop(g, ts, nf1, groundtruth=groundtruth, epoch=epoch)

        at = nf2[:, :4]
        slew = nf2[:, 4:8]
        # rat = nf2[:, 8:]

        return net_delays, cell_delays, at, slew  # , rat
