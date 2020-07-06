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
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

import torch as T
from torch import nn
from torch.nn import functional as F

from torch import Tensor
from ep_lstm_cell import EpLSTMCell

@dataclass
class EpLSTMCell_Builder:
    hidden_size                 : int
    vertical_dropout            : float = 0.0
    recurrent_dropout           : float = 0.0
    recurrent_dropout_mode      : str   = 'gal_tied'
    input_kernel_initialization : str   = 'xavier_uniform'
    recurrent_activation        : str   = 'sigmoid'
    tied_forget_gate            : bool  = False

    def make(self, input_size: int):
        return EpLSTMCell(input_size, self)

    def make_scripted(self, *p, **ks):
        return T.jit.script(self.make(*p, **ks))

class EpLSTM_Layer(nn.Module):
    def reorder_inputs(self, inputs: Union[List[T.Tensor], T.Tensor]):
        #^ inputs : [t b i]
        if self.direction == 'backward':
            return inputs[::-1]
        return inputs

    def __init__(
            self,
            cell: EpLSTMCell,
            direction='forward',
            batch_first=False,
    ):
        super().__init__()
        if isinstance(batch_first, bool):
            batch_first = (batch_first, batch_first)
        self.batch_first = batch_first
        self.direction = direction
        self.cell_: EpLSTMCell = cell

    @T.jit.ignore
    def forward(self, inputs, state_t0):
        x_t, m_t = inputs 
        if self.batch_first[0]:
        #^ x_t : [b t i]
            x_t = x_t.transpose(1, 0)
        #^ x_t : [t b i]
        x_t = x_t.unbind(0)

        if state_t0 is None:
            state_t0 = self.cell_.get_init_state(x_t)
    
        x_t = self.reorder_inputs(x_t)

        sequence, state = self.cell_.loop(x_t, m_t, state_t0)
   
        #^ sequence : t * [b h]
        sequence = self.reorder_inputs(sequence)
        sequence = T.stack(sequence)
        #^ sequence : [t b h]

        if self.batch_first[1]:
            sequence = sequence.transpose(1, 0)
        #^ sequence : [b t h]  

        return sequence, state


class EpLSTM(nn.Module):
    def __init__(
            self,
            input_size    : int,
            num_layers    : int,
            batch_first   : bool = False,
            scripted      : bool = True,
            *args, **kargs,
    ):
        super().__init__()
        self._cell_builder = EpLSTMCell_Builder(*args, **kargs)
    
        Dh = self._cell_builder.hidden_size
        def make(isize: int):
            cell = self._cell_builder.make_scripted(isize)
            return EpLSTM_Layer(cell, isize, batch_first=batch_first)

        rnns = [
            make(input_size),
            *[
                make(Dh)
                for _ in range(num_layers - 1)
            ],
        ]

        self.rnn = nn.Sequential(*rnns)

        self.input_size = input_size
        self.hidden_size = self._cell_builder.hidden_size
        self.num_layers = num_layers

    def __repr__(self):
        return (
            f'${self.__class__.__name__}'
            + '('
            + f'in={self.input_size}, '
            + f'hid={self.hidden_size}, '
            + f'layers={self.num_layers}, '
            + f'bi={self.bidirectional}'
            + '; '
            + str(self._cell_builder)
        )

    def forward(self, inputs, state_t0=None):
        for rnn in self.rnn:
            inputs, state = rnn(inputs, state_t0) 
        return inputs, state 

    def reset_parameters(self):
        for rnn in self.rnn:
            rnn.cell_.reset_parameters_()