from collections import defaultdict
from typing import List, Union, Dict
import dgl
import dgl.function as fn
import dgl.nn as dglnn

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
import sar
from typing import Dict

class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    norm : str, optional
        Normalization Method. Default: None
    """

    def __init__(
        self,
        in_feat,
        out_feat,
        rel_names,
        num_bases,
        *,
        weight=True,
        bias=True,
        activation=None,
        self_loop=False,
        dropout=0.0,
        num_ffn_layers_in_gnn=0,
        ffn_activation=F.relu,
        norm=None
    ):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(in_feat, out_feat, norm="right", weight=False, bias=False)
                for rel in rel_names
            }
        )

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis(
                    (in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(th.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))

        # get the node types
        ntypes = set()
        for rel in rel_names:
            ntypes.add(rel[0])
            ntypes.add(rel[2])
            
        # normalization
        self.norm = None
        if activation is None and norm is not None:
            raise ValueError("Cannot set gnn norm layer when activation layer is None")
        if norm == "batch":
            self.norm = nn.ParameterDict({ntype:nn.BatchNorm1d(out_feat) for ntype in ntypes})
        elif norm == "layer":
            self.norm = nn.ParameterDict({ntype:nn.LayerNorm(out_feat) for ntype in ntypes})
        else:
            # by default we don't apply any normalization
            self.norm = None
        
        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain("relu"))

        # ngnn
        self.num_ffn_layers_in_gnn = num_ffn_layers_in_gnn
        self.ngnn_mlp = NGNNMLP(out_feat, out_feat,
                                     num_ffn_layers_in_gnn, ffn_activation, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation

        Parameters
        ----------
        g : DGLGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.

        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        with g.local_scope():
            if self.use_weight:
                weight = self.basis() if self.use_basis else self.weight
                wdict = {self.rel_names[i]: {"weight": w.squeeze(0)} \
                    for i, w in enumerate(th.split(weight, 1, dim=0))}
            else:
                wdict = {}

            if g.is_block:
                inputs_src = inputs
                inputs_dst = {}
                for k in g.dsttypes:
                    # If the destination node type exists in the input embeddings,
                    # we can get from the input node embeddings directly because
                    # the input nodes of DGL's block also contain the destination nodes
                    if k in inputs:
                        inputs_dst[k] = inputs[k][:g.number_of_dst_nodes(k)]
                    else:
                        # If the destination node type doesn't exist (this may happen if
                        # we use RGCN to construct node features), we should create a zero
                        # tensor. This tensor won't be used for computing embeddings.
                        # We need this just to fulfill the requirements of DGL message passing
                        # modules.
                        if g.num_dst_nodes(k) > 0:
                            assert not self.self_loop, \
                                    f"We cannot allow self-loop if node {k} doesn't have input features."
                        inputs_dst[k] = th.zeros((g.num_dst_nodes(k), self.in_feat),
                                                dtype=th.float32, device="cpu")
                
                # inputs_dst = {
                #     k: v[: g.number_of_dst_nodes(k)] for k, v in inputs.items()
                # }
            else:
                inputs_src = inputs_dst = inputs

            hs = self.conv(g, (inputs_src, inputs_dst), mod_kwargs=wdict)
            # hs = self.conv(g, inputs, mod_kwargs=wdict)

            def _apply(ntype, h):
                if self.self_loop:
                    h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
                if self.bias:
                    h = h + self.h_bias
                if self.norm:
                    h = self.norm[ntype](h)
                if self.activation:
                    h = self.activation(h)
                if self.num_ffn_layers_in_gnn > 0:
                    h = self.ngnn_mlp(h)
                return self.dropout(h)
            
            for k, _ in inputs.items():
                if g.number_of_dst_nodes(k) > 0:
                    if k not in hs:
                        hs[k] = th.zeros((g.number_of_dst_nodes(k),
                                      self.out_feat), device=inputs[k].device)
                        # TODO the above might fail if the device is a different GPU

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}
    
    
class RelGraphEmbed(nn.Module):
    """The input encoder layer"""

    def __init__(
        self,
        g,
        in_dim,
        embed_size,
        activation=None,
        dropout=0.0,
        use_node_embeddings=False,
        force_no_embeddings=None,
        num_ffn_layers_in_input=0,
        ffn_activation=F.relu
    ):
        super(RelGraphEmbed, self).__init__()
        self.g = g
        self.embed_size = embed_size
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.learnable_embeddings = {}
        self.use_node_embeddings = use_node_embeddings
        if force_no_embeddings is None:
            force_no_embeddings = []
            
        # for ogbn-mag dataset
        feat_size = {}
        for ntype in g.ntypes_global:
            if ntype == "paper":
                feat_size[ntype] = in_dim
            else:
                feat_size[ntype] = 0

        # create weight embeddings for each node for each relation
        self.proj_matrix = nn.ParameterDict()
        self.input_projs = nn.ParameterDict()
        for ntype in g.ntypes_global:
            feat_dim = 0
            if feat_size[ntype] > 0:
                feat_dim += feat_size[ntype]
            if feat_dim > 0:
                input_projs = nn.Parameter(th.Tensor(feat_dim, self.embed_size))
                nn.init.xavier_uniform_(input_projs, gain=nn.init.calculate_gain('relu'))
                self.input_projs[ntype] = input_projs
                if self.use_node_embeddings:
                    self.learnable_embeddings[ntype] = nn.Embedding(g.num_src_nodes(ntype), self.embed_size)
                    proj_matrix = nn.Parameter(th.Tensor(2 * self.embed_size, self.embed_size))
                    nn.init.xavier_uniform_(proj_matrix, gain=nn.init.calculate_gain('relu'))
                    # nn.ParameterDict support this assignment operation if not None,
                    # so disable the pylint error
                    self.proj_matrix[ntype] = proj_matrix   # pylint: disable=unsupported-assignment-operation
            elif ntype not in force_no_embeddings:
                proj_matrix = nn.Parameter(th.Tensor(self.embed_size, self.embed_size))
                nn.init.xavier_uniform_(proj_matrix, gain=nn.init.calculate_gain('relu'))
                self.proj_matrix[ntype] = proj_matrix
                self.learnable_embeddings[ntype] = nn.Embedding(g.num_src_nodes(ntype), self.embed_size)
        
        self.num_ffn_layers_in_input = num_ffn_layers_in_input
        self.ngnn_mlp = nn.ModuleDict({})
        for ntype in g.ntypes:
            self.ngnn_mlp[ntype] = NGNNMLP(embed_size, embed_size,
                            num_ffn_layers_in_input, ffn_activation, dropout)

    def forward(self, input_feats, input_nodes):
        assert isinstance(input_feats, dict), 'The input features should be in a dict.'
        assert isinstance(input_nodes, dict), 'The input node IDs should be in a dict.'
        embs = {}
        for ntype in input_nodes:
            emb = None
            if ntype in input_feats:
                assert ntype in self.input_projs, \
                        f"We need a projection for node type {ntype}"
                # If the input data is not float, we need to convert it t float first.
                emb = input_feats[ntype].float() @ self.input_projs[ntype]
                if self.use_node_embeddings:
                    assert ntype in self.learnable_embeddings, \
                            f"We need sparse embedding for node type {ntype}"
                    # get node typ ids instead of homogenous ids
                    # _, node_type_ids = self.g.partition_book.map_to_per_ntype(input_nodes[ntype] + self.g.node_ranges[sar.comm.rank()][0])
                    node_emb = self.learnable_embeddings[ntype](input_nodes[ntype], emb.device)
                    concat_emb=th.cat((emb, node_emb),dim=1)
                    emb = concat_emb @ self.proj_matrix[ntype]
            elif ntype in self.learnable_embeddings: # nodes do not have input features
                # If the number of the input node of a node type is 0,
                # return an empty tensor with shape (0, emb_size)
                device = self.proj_matrix[ntype].device
                if len(input_nodes[ntype]) == 0:
                    dtype = self.learnable_embeddings[ntype].weight.dtype
                    embs[ntype] = th.zeros((0, self.learnable_embeddings[ntype].embedding_dim),
                                           device=device, dtype=dtype)
                    continue
                # _, node_type_ids = self.g.partition_book.map_to_per_ntype(input_nodes[ntype] + self.g.node_ranges[sar.comm.rank()][0])
                emb = self.learnable_embeddings[ntype](input_nodes[ntype]).to(device)
                emb = emb @ self.proj_matrix[ntype]
            if emb is not None:
                if self.activation is not None:
                    emb = self.activation(emb)
                    emb = self.dropout(emb)
                embs[ntype] = emb

        def _apply(t, h):
            if self.num_ffn_layers_in_input > 0:
                h = self.ngnn_mlp[t](h)
            return h

        embs = {ntype: _apply(ntype, h) for ntype, h in embs.items()}
        return embs
    
    def get_sparse_params(self):
        """ get the sparse parameters.

        Returns
        -------
        list of Tensors: the sparse embeddings.
        """
        if self.learnable_embeddings is not None and len(self.learnable_embeddings) > 0:
            return list(self.learnable_embeddings.values())
        else:
            return []
    
    @property
    def out_dims(self):
        """ The number of output dimensions.
        """
        return self.embed_size

  
class NGNNMLP(nn.Module):
    r"""NGNN MLP Implementation

    NGNN Layer is consisted of combination of a MLP Layer, an activation layer and dropout

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    layer_number: int
        Number of NGNN layers
    activation: torch.nn.functional
        Type of NGNN activation layer
    dropout : float, optional
        Dropout rate. Default: 0.0
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 layer_number=0,
                 activation=F.relu,
                 dropout=0.0):
        super(NGNNMLP, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.activation = activation
        self.layer_number = 0
        self.dropout = nn.Dropout(dropout)
        self.ngnn_gnn = nn.ParameterList()
        for _ in range(0, layer_number):
            mlp_layer = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(mlp_layer, gain=nn.init.calculate_gain('relu'))
            self.ngnn_gnn.append(mlp_layer)

    # pylint: disable=invalid-name
    def forward(self, emb):
        """Forward computation

        Parameters
        ----------
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        h = emb
        for layer in self.ngnn_gnn:
            h = th.matmul(h, layer)
            h = self.activation(h)
        return self.dropout(h)
  
    
class RGCNEncoder(nn.Module):
    def __init__(
        self,
        g,
        in_dim,
        h_dim,
        out_dim,
        num_labels,
        num_bases=-1,
        num_hidden_layers=1,
        dropout=0,
        use_self_loop=True,
        last_layer_act=False,
        num_ffn_layers_in_gnn=0,
        norm=None
    ):
        super(RGCNEncoder, self).__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        if num_bases < 0 or num_bases > len(g.canonical_etypes_global):
            self.num_bases = len(g.canonical_etypes_global)
        else:
            self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.num_labels = num_labels

        self.embed_layer = RelGraphEmbed(g, in_dim, self.h_dim)
        self.layers = nn.ModuleList()

        # h2h
        for _ in range(self.num_hidden_layers):
            self.layers.append(
                RelGraphConvLayer(
                    h_dim,
                    h_dim,
                    g.canonical_etypes_global,
                    self.num_bases,
                    activation=F.relu,
                    self_loop=use_self_loop,
                    dropout=dropout,
                    num_ffn_layers_in_gnn=num_ffn_layers_in_gnn,
                    ffn_activation=F.relu,
                    norm=norm
                )
            )
        # h2o
        self.layers.append(
            RelGraphConvLayer(
                h_dim,
                out_dim,
                g.canonical_etypes_global,
                self.num_bases,
                activation=F.relu if last_layer_act else None,
                self_loop=use_self_loop,
                norm=norm if last_layer_act else None
            )
        )
        
        self.decoder = Decoder(out_dim, num_labels, False)
        self.loss_fn = nn.CrossEntropyLoss(weight=None)
        
    def forward(self,
                blocks: List[Union[sar.GraphShardManager, sar.DistributedBlock]],
                features: Dict[str, th.Tensor],
                input_nodes: Dict[str, th.Tensor]):
        """
        Forward computation

        Parameters
        ----------
        blocks: List[Union[sar.GraphShardManager, sar.DistributedBlock]]
            Created MFGs blocks
        features: Dict[str, torch.Tensor]
            Input node feature for each node type.
        input_nodes: Dict[str, torch.Tensor]
            Input indices of each node type
        """
        h = self.embed_layer(features, input_nodes)
        for idx, layer in enumerate(self.layers):
            h = layer(blocks[idx], h)
        return h


class Decoder(nn.Module):
    def __init__(self,
                 in_dim,
                 num_classes,
                 multilabel,
                 dropout=0):
        super(Decoder, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.multilabel = multilabel
        self.decoder = nn.Parameter(th.Tensor(in_dim, num_classes))
        nn.init.xavier_uniform_(self.decoder,
                                gain=nn.init.calculate_gain('relu'))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        return th.matmul(inputs, self.decoder)
    
    def predict(self, inputs):
        logits = th.matmul(inputs, self.decoder)
        return (th.sigmoid(logits) > .5).long() if self._multilabel else logits.argmax(dim=1)
    
    def predict_proba(self, inputs):
        logits = th.matmul(inputs, self.decoder)
        return th.sigmoid(logits) if self._multilabel else th.softmax(logits, 1)
