import os

import torch
import torch.nn as nn

from espnet2.asr.transducer.transducer_decoder import TransducerDecoder

from espnet_onnx.utils.function import subsequent_mask
from ..language_models.lm import Embedding
from ..abs_model import AbsModel


class TransducerDecoder(nn.Module, AbsModel):
    def __init__(self, model):
        super().__init__()
        self.embed = Embedding(model.embed)
        self.decoder = model.decoder
        self.dlayers = model.dlayers
        self.dunits = model.dunits
        self.dtype = model.dtype

    def forward(self, labels, h_cache, c_cache):
        # embed and rnn-forward
        sequence = self.embed(labels)
        h_next = torch.zeros(self.dlayers, h_cache.size(1), self.dunits)
        c_next = torch.zeros(self.dlayers, c_cache.size(1), self.dunits)
        if self.dtype == "lstm":
            for i in range(self.dlayers):
                sequence, (h_next[i:i+1], c_next[i:i+1]) = self.decoder[i](
                    sequence,
                    hx=(h_cache[i:i+1], c_cache[i:i+1]) 
                )
        else:
            for i in range(self.dlayers):
                sequence, h_next[i:i+1] = self.decoder[i](
                    sequence, hx=h_cache[i:i+1]
                )
        return sequence, h_next, c_next

    def get_dummy_inputs(self, enc_size):
        labels = torch.LongTensor([0, 1]).unsqueeze(0)
        h_cache = torch.randn(self.dlayers, 1, self.dunits)
        c_cache = torch.randn(self.dlayers, 1, self.dunits)
        return labels, h_cache, c_cache

    def get_input_names(self):
        return ['labels', 'h_cache', 'c_cache'] 

    def get_output_names(self):
        return ['sequence', 'out_h_cache', 'out_c_cache_']
            
    def get_dynamic_axes(self):
        ret = {
            'labels': {
                0: 'labels_batch',
                1: 'labels_length'
            }
        }
        ret.update({
            f'h_cache': {
                1: f'h_cache_length'
            },
            f'c_cache': {
                1: f'c_cache_length'
            },
            f'out_h_cache': {
                1: f'out_h_cache_length'
            },
            f'out_c_cache': {
                1: f'out_c_cache_length'
            }
        })
        return ret

    def get_model_config(self, path):
        file_name = os.path.join(path, 'decoder.onnx')
        return {
            "dec_type": "TransducerDecoder",
            "model_path": file_name,
            "n_layers": self.dlayers,
            "odim": self.dunits,
            "dtype": self.dtype
        }
