from typing import Any
from ...model import bracket_net

from .util import NaiveBase, NNAstarLikeBase

class ModelInterface():
    def __init__(self, Model, mode):
        self.mode = mode
        self.model = Model

    def __call__(self, d_vocab, pos_encoder, d_model=128, n_head=4,
                 num_layers=6, dropout=0.0,
                 embed=None):
        return self.model(d_vocab, pos_encoder,
                          d_model=d_model, n_head=n_head,
                          num_layers=num_layers, dropout=dropout,
                          embed=embed, mode=self.mode)


class Naive(NaiveBase):
    def __init__(self, config):
        super().__init__(config,
                         ModelInterface(
                             bracket_net.BracketNet, config.model.mode))


class NNAstarLike(NNAstarLikeBase):
    def __init__(self, config):
        super().__init__(config,
                         ModelInterface(
                             bracket_net.BracketNet, config.model.mode))