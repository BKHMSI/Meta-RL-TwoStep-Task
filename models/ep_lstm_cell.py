from typing import (
    Tuple,
    List,
    Optional,
    Dict,
    Callable,
    Union,
    cast,
)
from collections import namedtuple
from dataclasses import dataclass

import numpy as np

import torch as T
from torch import nn
from torch import Tensor
from torch.nn import functional as F

# constants 
N_GATES = 5
GateSpans = namedtuple('GateSpans', ['I', 'F', 'G', 'O', 'R'])

ACTIVATIONS = {
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'hard_tanh': nn.Hardtanh(),
    'relu': nn.ReLU(),
}


class EpLSTMCell:
    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            + ', '.join(
                [
                    f'in: {self.Dx}',
                    f'hid: {self.Dh}',
                    f'rdo: {self.recurrent_dropout_p} @{self.recurrent_dropout_mode}',
                    f'vdo: {self.vertical_dropout_p}'
                ]
            )
            +')'
        )

    def __init__(
            self,
            input_size: int,
            args: EpLSTMCell_Builder,
    ):
        super().__init__()
        self._args = args
        self.Dx = input_size
        self.Dh = args.hidden_size
        self.recurrent_kernel = nn.Linear(self.Dh, self.Dh * N_GATES)
        self.input_kernel     = nn.Linear(self.Dx, self.Dh * N_GATES)

        self.recurrent_dropout_p    = args.recurrent_dropout or 0.0
        self.vertical_dropout_p     = args.vertical_dropout or 0.0
        self.recurrent_dropout_mode = args.recurrent_dropout_mode
        
        self.recurrent_dropout = nn.Dropout(self.recurrent_dropout_p)
        self.vertical_dropout  = nn.Dropout(self.vertical_dropout_p)

        self.tied_forget_gate = args.tied_forget_gate

        if isinstance(args.recurrent_activation, str):
            self.fun_rec = ACTIVATIONS[args.recurrent_activation]
        else:
            self.fun_rec = args.recurrent_activation

        self.reset_parameters_()

    # @T.jit.ignore
    def get_recurrent_weights(self):
        # type: () -> Tuple[GateSpans, GateSpans]
        W = self.recurrent_kernel.weight.chunk(N_GATES, 0)
        b = self.recurrent_kernel.bias.chunk(N_GATES, 0)
        W = GateSpans(W[0], W[1], W[2], W[3], W[4])
        b = GateSpans(b[0], b[1], b[2], b[3], b[4])
        return W, b

    # @T.jit.ignore
    def get_input_weights(self):
        # type: () -> Tuple[GateSpans, GateSpans]
        W = self.input_kernel.weight.chunk(N_GATES, 0)
        b = self.input_kernel.bias.chunk(N_GATES, 0)
        W = GateSpans(W[0], W[1], W[2], W[3], W[4])
        b = GateSpans(b[0], b[1], b[2], b[3], b[4])
        return W, b

    @T.jit.ignore
    def reset_parameters_(self):
        rw, rb = self.get_recurrent_weights()
        iw, ib = self.get_input_weights()

        nn.init.zeros_(self.input_kernel.bias)
        nn.init.zeros_(self.recurrent_kernel.bias)
        nn.init.ones_(rb.F)
        #^ forget bias

        for W in rw:
            nn.init.orthogonal_(W)
        for W in iw:
            nn.init.xavier_uniform_(W)

    @T.jit.export
    def get_init_state(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = input.shape[1]
        zeros = T.zeros(batch_size, self.Dh, device=input.device)
        return (zeros, zeros)

    def apply_input_kernel(self, xt: Tensor) -> List[Tensor]:
        xto = self.vertical_dropout(xt)
        out = self.input_kernel(xto).chunk(N_GATES, 1)
        return out

    def apply_recurrent_kernel(self, h_tm1: Tensor):
        #^ h_tm1 : [b h]
        mode = self.recurrent_dropout_mode
        if mode == 'gal_tied':
            hto = self.recurrent_dropout(h_tm1)
            out = self.recurrent_kernel(hto)
            #^ out : [b N_GATES*h]
            outs = out.chunk(N_GATES, -1)
        elif mode == 'gal_gates':
            outs = []
            WW, bb = self.get_recurrent_weights()
            for i in range(N_GATES):
                hto = self.recurrent_dropout(h_tm1)
                outs.append(F.linear(hto, WW[i], bb[i]))
        else:
            outs = self.recurrent_kernel(h_tm1).chunk(N_GATES, -1)
        return outs

    def forward(self, inputs, state):
        # type: (Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        #^ inputs.xt : [b i]
        #^ state.h : [b h]

        (x_t, mem_t) = inputs 
        (h_tm1, c_tm1) = state

        Xi, Xf, Xg, Xo, Xr = self.apply_input_kernel(x_t)
        Hi, Hf, Hg, Ho, Hr = self.apply_recurrent_kernel(h_tm1)

        ft = self.fun_rec(Xf + Hf)
        ot = self.fun_rec(Xo + Ho)
        if self.tied_forget_gate:
            it = 1.0 - ft
        else:
            it = self.fun_rec(Xi + Hi)

        gt = T.tanh(Xg + Hg) # * np.sqrt(3)
        if self.recurrent_dropout_mode == 'semeniuta':
            #* https://arxiv.org/abs/1603.05118
            gt = self.recurrent_dropout(gt)

        rt = self.fun_rec(Xr + Hr)

        ct = (ft * c_tm1) + (it * gt) + (rt * T.tanh(mem_t))

        ht = ot * T.tanh(ct)

        return ht, (ht, ct)

    @T.jit.export
    def loop(self, inputs, state_t0, mask=None):
        # type: (List[Tensor], Tuple[Tensor, Tensor], Optional[List[Tensor]]) -> Tuple[List[Tensor], Tuple[Tensor, Tensor]]
        '''
        This loops over t (time) steps
        '''
        #^ inputs      : t * [b i]
        #^ state_t0[i] : [b s]
        #^ out         : [t b h]
        state = state_t0
        outs = []
        for xt in inputs:
            ht, state = self(xt, state)
            outs.append(ht)

        return outs, state

