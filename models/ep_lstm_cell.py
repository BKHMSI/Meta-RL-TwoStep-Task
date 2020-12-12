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

# from models.ep_lstm import EpLSTMCell_Builder

# constants 
N_GATES = 5
GateSpans = namedtuple('GateSpans', ['I', 'F', 'G', 'O', 'R'])

ACTIVATIONS = {
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'hard_tanh': nn.Hardtanh(),
    'relu': nn.ReLU(),
}

class EpLSTMCell(nn.Module):
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
            input_size                  : int,
            hidden_size                 : int,
            vertical_dropout            : float = 0.0,
            recurrent_dropout           : float = 0.0,
            recurrent_dropout_mode      : str   = 'gal_tied',
            recurrent_activation        : str   = 'sigmoid',
            tied_forget_gate            : bool  = False,
            noise_idx                   : list  = None,
            noise_level                 : int   = 0

    ):
        super().__init__()
        self.Dx = input_size
        self.Dh = hidden_size
        self.recurrent_kernel = nn.Linear(self.Dh, self.Dh * N_GATES)
        self.input_kernel     = nn.Linear(self.Dx, self.Dh * N_GATES)

        self.recurrent_dropout_p    = recurrent_dropout or 0.0
        self.vertical_dropout_p     = vertical_dropout or 0.0
        self.recurrent_dropout_mode = recurrent_dropout_mode
        
        self.recurrent_dropout = nn.Dropout(self.recurrent_dropout_p)
        self.vertical_dropout  = nn.Dropout(self.vertical_dropout_p)

        self.tied_forget_gate = tied_forget_gate

        if isinstance(recurrent_activation, str):
            self.fun_rec = ACTIVATIONS[recurrent_activation]
        else:
            self.fun_rec = recurrent_activation

        self.noise_idx = noise_idx
        self.noise_level = noise_level

        self.reset_parameters_()
        self.reset_gate_monitor()


    # @T.jit.ignore
    def get_recurrent_weights(self):
        # type: () -> Tuple[GateSpans, GateSpans]
        W = self.recurrent_kernel.weight.chunk(5, 0)
        b = self.recurrent_kernel.bias.chunk(5, 0)
        W = GateSpans(W[0], W[1], W[2], W[3], W[4])
        b = GateSpans(b[0], b[1], b[2], b[3], b[4])
        return W, b

    # @T.jit.ignore
    def get_input_weights(self):
        # type: () -> Tuple[GateSpans, GateSpans]
        W = self.input_kernel.weight.chunk(5, 0)
        b = self.input_kernel.bias.chunk(5, 0)
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
        out = self.input_kernel(xto).chunk(5, 1)
        return out

    def apply_recurrent_kernel(self, h_tm1: Tensor):
        #^ h_tm1 : [b h]
        mode = self.recurrent_dropout_mode
        if mode == 'gal_tied':
            hto = self.recurrent_dropout(h_tm1)
            out = self.recurrent_kernel(hto)
            #^ out : [b 5*h]
            outs = out.chunk(5, -1)
        elif mode == 'gal_gates':
            outs = []
            WW, bb = self.get_recurrent_weights()
            for i in range(5):
                hto = self.recurrent_dropout(h_tm1)
                outs.append(F.linear(hto, WW[i], bb[i]))
        else:
            outs = self.recurrent_kernel(h_tm1).chunk(5, -1)
        return outs

    def forward(self, xt, mt, state):
        # type: (Tensor, Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        #^ inputs.xt : [b i]
        #^ state.h : [b h]

        (h_tm1, c_tm1) = state

        Xi, Xf, Xg, Xo, Xr = self.apply_input_kernel(xt)
        Hi, Hf, Hg, Ho, Hr = self.apply_recurrent_kernel(h_tm1)

        ft = self.fun_rec(Xf + Hf)
        ot = self.fun_rec(Xo + Ho)
        if self.tied_forget_gate:
            it = 1.0 - ft
        else:
            it = self.fun_rec(Xi + Hi)

        gt = T.tanh(Xg + Hg) 
        if self.recurrent_dropout_mode == 'semeniuta':
            #* https://arxiv.org/abs/1603.05118
            gt = self.recurrent_dropout(gt)

        rt = self.fun_rec(Xr + Hr)

        self.rt_gate += [rt.view(-1).detach().cpu().numpy()]
        self.it_gate += [it.view(-1).detach().cpu().numpy()]
        self.ft_gate += [ft.view(-1).detach().cpu().numpy()]

        ct = (ft * c_tm1) + (it * gt) + (rt * T.tanh(mt))

        # if self.noise_idx is not None and self.noise_level > 0:
            # noise = T.randn(len(self.noise_idx))
            # ct[:, self.noise_idx] = ct[:, self.noise_idx] + noise * self.noise_level
            # ct[:, self.noise_idx] = 0

        ht = ot * T.tanh(ct)

        return ht, (ht, ct)

    def reset_gate_monitor(self):
        self.rt_gate = []
        self.it_gate = []
        self.ft_gate = []
