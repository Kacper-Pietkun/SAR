# Copyright (c) 2022 Intel Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import List, Union, Dict
from argparse import ArgumentParser
import os
import logging
import time
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import dgl  # type: ignore
from model import RGCNEncoder
import sar


parser = ArgumentParser(
    description="Script tailored to GraphStorm flow for benchmarking")


parser.add_argument("--partitioning-json-file", type=str, default="",
                    help="Path to the .json file containing partitioning information")

parser.add_argument('--ip-file', default='./ip_file', type=str,
                    help='File with ip-address. Worker 0 creates this file and all others read it')

parser.add_argument('--backend', default='nccl', type=str, choices=['ccl', 'nccl', 'mpi', 'gloo'],
                    help='Communication backend to use')


parser.add_argument('--rank', default=0, type=int,
                    help='Rank of the current worker ')

parser.add_argument('--world-size', default=2, type=int,
                    help='Number of workers ')


parser.add_argument("--cpu-run", action="store_true",
                    help="Run on CPUs if set, otherwise run on GPUs ")


parser.add_argument('--n-layers', default=2, type=int,
                    help='Number of GNN layers ')

parser.add_argument('--hidden-layer-dim', default=256, type=int,
                    help='Dimension of GNN hidden layer')

parser.add_argument("--lr", type=float, default=1e-2,
                    help="learning rate")


parser.add_argument('--target-node-type', default="paper", type=str,
                    help='Target node for prediction')

parser.add_argument('--label-field', default="labels", type=str,
                    help='Name of the field that stores labels')

parser.add_argument('--feat-field', default="feat", type=str,
                    help='Name of the field that stores features')


def main():
    args = parser.parse_args()
    print('args', args)

    sar.patch_dgl()
    
    use_gpu = torch.cuda.is_available() and not args.cpu_run
    device = torch.device('cuda' if use_gpu else 'cpu')

    # Obtain the ip address of the master through the network file system
    master_ip_address = sar.nfs_ip_init(args.rank, args.ip_file)
    sar.initialize_comms(args.rank, args.world_size, master_ip_address, args.backend)

    # Load DGL partition data
    partition_data = sar.load_dgl_partition_data(args.partitioning_json_file, args.rank, device)
    partition_data.node_features["paper/feat"] = partition_data.node_features["paper/feat"].double()
    
    bool_masks = {}
    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        local_mask = sar.suffix_key_lookup(partition_data.node_features,
                                           mask_name,
                                           expand_to_all = False,
                                           type_list = partition_data.node_type_names)
        bool_masks[mask_name] = local_mask.bool()

    indices_masks = {}
    for mask_name, indices_name in zip(['train_mask', 'val_mask', 'test_mask'],
                                       ['train_indices', 'val_indices', 'test_indices']):
        global_mask = sar.suffix_key_lookup(partition_data.node_features,
                                            mask_name,
                                            expand_to_all = True,
                                            type_list = partition_data.node_type_names)
        indices_masks[indices_name] = global_mask.nonzero(as_tuple=False).view(-1).to(device)

    labels = sar.suffix_key_lookup(partition_data.node_features,
                                   args.label_field,
                                   expand_to_all = False,
                                   type_list = partition_data.node_type_names).long()
    labels.to(device)
    # Obtain the number of classes by finding the max label across all workers
    num_labels = labels.max() + 1
    sar.comm.all_reduce(num_labels, dist.ReduceOp.MAX, move_to_comm_device=True)
    num_labels = num_labels.item()
    
    # Get features
    features = sar.suffix_key_lookup(partition_data.node_features,
                                     args.feat_field,
                                     type_list = partition_data.node_type_names                                     
                                     ).to(device)
    
    # Create MFGs
    eval_blocks = sar.construct_mfgs(partition_data,
                                     torch.cat((indices_masks['train_indices'],
                                                indices_masks['val_indices'],
                                                indices_masks['test_indices'])) +
                                     partition_data.node_ranges[sar.comm.rank()][0],
                                     args.n_layers)
    eval_blocks = [block.to(device) for block in eval_blocks]
   
    # Create a GNN model
    gnn_model = RGCNEncoder(eval_blocks[0],
                            features.shape[1],
                            args.hidden_layer_dim,
                            args.hidden_layer_dim,
                            num_labels).to(device)
    print('model', gnn_model)
    
    # Synchronize the model parmeters across all workers
    sar.sync_params(gnn_model)
    
    features = {args.target_node_type: features[eval_blocks[0].srcnodes(args.target_node_type)]}
    
    # We do not need the partition data anymore
    del partition_data
    
    input_nodes = {type: eval_blocks[0].srcnodes(type) for type in eval_blocks[0].srctypes}
    # inference
    gnn_model.eval()
    with torch.no_grad():
        logits = gnn_model(eval_blocks, features, input_nodes)
    print("DONE")


if __name__ == '__main__':
    main()
