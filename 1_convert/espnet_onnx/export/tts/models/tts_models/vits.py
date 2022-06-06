
from typing import (
    Optional,
    Tuple
)

import math
import torch
import torch.nn as nn

from espnet_onnx.export.asr.models.language_models.lm import get_pos_emb
from espnet_onnx.utils.torch_function import make_pad_mask

from ..abs_model import AbsModel


class OnnxTextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, x_lengths):
        x = self.model.emb(x) * math.sqrt(self.model.attention_dim)
        x_mask = 1 - make_pad_mask(x_lengths).unsqueeze(1).type(torch.float32)
        # encoder assume the channel last (B, T_text, attention_dim)
        # but mask shape shoud be (B, 1, T_text)
        x, _ = self.model.encoder(x, x_mask)

        # convert the channel first (B, attention_dim, T_text)
        x = x.transpose(1, 2)
        stats = self.model.proj(x) * x_mask
        m, logs = stats.split(stats.size(1) // 2, dim=1)
        return x, m, logs, x_mask


class OnnxVITSGenerator(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text_encoder = OnnxTextEncoder(model.text_encoder)
        self.decoder = model.decoder
        self.posterior_encoder = model.posterior_encoder
        self.flow = model.flow
        self.duration_predictor = model.duration_predictor
        self.model = model

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        feats_lengths: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        dur: Optional[torch.Tensor] = None,
        noise_scale: float = 0.667,
        noise_scale_dur: float = 0.8,
        alpha: float = 1.0,
        max_len: Optional[int] = None,
        use_teacher_forcing: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run inference.

        Args:
            text (Tensor): Input text index tensor (B, T_text,).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, aux_channels, T_feats,).
            feats_lengths (Tensor): Feature length tensor (B,).
            sids (Optional[Tensor]): Speaker index tensor (B,) or (B, 1).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, spk_embed_dim).
            lids (Optional[Tensor]): Language index tensor (B,) or (B, 1).
            dur (Optional[Tensor]): Ground-truth duration (B, T_text,). If provided,
                skip the prediction of durations (i.e., teacher forcing).
            noise_scale (float): Noise scale parameter for flow.
            noise_scale_dur (float): Noise scale parameter for duration predictor.
            alpha (float): Alpha parameter to control the speed of generated speech.
            max_len (Optional[int]): Maximum length of acoustic feature sequence.
            use_teacher_forcing (bool): Whether to use teacher forcing.

        Returns:
            Tensor: Generated waveform tensor (B, T_wav).
            Tensor: Monotonic attention weight tensor (B, T_feats, T_text).
            Tensor: Duration tensor (B, T_text).

        """
        # encoder
        x, m_p, logs_p, x_mask = self.text_encoder(text, text_lengths)
        g = None
        if self.model.spks is not None:
            # (B, global_channels, 1)
            g = self.model.global_emb(sids.view(-1)).unsqueeze(-1)
        if self.model.spk_embed_dim is not None:
            # (B, global_channels, 1)
            g_ = self.model.spemb_proj(F.normalize(
                spembs.unsqueeze(0))).unsqueeze(-1)
            if g is None:
                g = g_
            else:
                g = g + g_
        if self.model.langs is not None:
            # (B, global_channels, 1)
            g_ = self.model.lang_emb(lids.view(-1)).unsqueeze(-1)
            if g is None:
                g = g_
            else:
                g = g + g_

        if use_teacher_forcing:
            # forward posterior encoder
            z, m_q, logs_q, y_mask = self.posterior_encoder(
                feats, feats_lengths, g=g)

            # forward flow
            z_p = self.flow(z, y_mask, g=g)  # (B, H, T_feats)

            # monotonic alignment search
            s_p_sq_r = torch.exp(-2 * logs_p)  # (B, H, T_text)
            # (B, 1, T_text)
            neg_x_ent_1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p,
                [1],
                keepdim=True,
            )
            # (B, T_feats, H) x (B, H, T_text) = (B, T_feats, T_text)
            neg_x_ent_2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2),
                s_p_sq_r,
            )
            # (B, T_feats, H) x (B, H, T_text) = (B, T_feats, T_text)
            neg_x_ent_3 = torch.matmul(
                z_p.transpose(1, 2),
                (m_p * s_p_sq_r),
            )
            # (B, 1, T_text)
            neg_x_ent_4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r,
                [1],
                keepdim=True,
            )
            # (B, T_feats, T_text)
            neg_x_ent = neg_x_ent_1 + neg_x_ent_2 + neg_x_ent_3 + neg_x_ent_4
            # (B, 1, T_feats, T_text)
            attn_mask = torch.unsqueeze(
                x_mask, 2) * torch.unsqueeze(y_mask, -1)
            # monotonic attention weight: (B, 1, T_feats, T_text)
            attn = self.model.maximum_path(
                neg_x_ent,
                attn_mask.squeeze(1),
            ).unsqueeze(1)
            dur = attn.sum(2)  # (B, 1, T_text)

            # forward decoder with random segments
            wav = self.decoder(z * y_mask, g=g)
        else:
            # duration
            if dur is None:
                logw = self.duration_predictor(
                    x,
                    x_mask,
                    g=g,
                    inverse=True,
                    noise_scale=noise_scale_dur,
                    text_lengths=text_lengths[0],
                )
                w = torch.exp(logw) * x_mask * alpha
                dur = torch.ceil(w)
            y_lengths = torch.clamp_min(torch.sum(dur, [1, 2]), 1).long()
            # y_mask = 1 - \
            #     make_pad_mask(y_lengths).unsqueeze(1).type(torch.float32)
            y_mask = 1 - make_pad_mask(torch.Tensor([512]).long()).unsqueeze(1).type(torch.float32)
            attn_mask = torch.unsqueeze(
                x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = self.model._generate_path(dur, attn_mask)

            # expand the length to match with the feature sequence
            # (B, T_feats, T_text) x (B, T_text, H) -> (B, H, T_feats)
            m_p = torch.matmul(
                attn.squeeze(1),
                m_p.transpose(1, 2),
            ).transpose(1, 2)
            # (B, T_feats, T_text) x (B, T_text, H) -> (B, H, T_feats)
            logs_p = torch.matmul(
                attn.squeeze(1),
                logs_p.transpose(1, 2),
            ).transpose(1, 2)

            # decoder
            # z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
            z_p = m_p
            z = self.flow(z_p, y_mask, g=g, inverse=True)
            wav = self.decoder((z * y_mask)[:, :, :max_len], g=g)

        return wav.squeeze(1), attn.squeeze(1), dur.squeeze(1), 256*y_lengths


class OnnxVITSModel(nn.Module, AbsModel):
    def __init__(
        self,
        model,
        noise_scale: float = 0.667,
        noise_scale_dur: float = 0.8,
        alpha: float = 1.0,
        max_len: int = None,
        use_teacher_forcing: bool = False,
        predict_duration: bool = False
    ):
        super().__init__()
        self.model = model
        self.generator = OnnxVITSGenerator(model.generator)
        self.use_teacher_forcing = use_teacher_forcing
        self.noise_scale = noise_scale
        self.noise_scale_dur = noise_scale_dur
        self.alpha = alpha
        self.max_len = max_len
        self.predict_duration = predict_duration

        # fix RelPositionalEncoding
        self.model.generator.text_encoder.encoder.embed[0] = \
            get_pos_emb(self.model.generator.text_encoder.encoder.embed[0])

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        feats_lengths: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
    ):
        # setup
        text = text[None]

        if self.use_teacher_forcing:
            assert feats is not None
            feats = feats[None].transpose(1, 2)
            wav, att_w, dur = self.generator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
                max_len=self.max_len,
                use_teacher_forcing=self.use_teacher_forcing,
            )
        else:
            wav, att_w, dur, y_length = self.generator(
                text=text,
                text_lengths=text_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
                dur=durations,
                noise_scale=self.noise_scale,
                noise_scale_dur=self.noise_scale_dur,
                alpha=self.alpha,
                max_len=self.max_len,
            )
        return dict(wav=wav.view(-1), att_w=att_w[0], duration=dur[0], y_length=y_length)

    def get_dummy_inputs(self):
        text = torch.LongTensor([32,18,14,13,10,6,32,4,2,5,4,28,17,66,7,4,12,6,2,17,47,4,13,3,9,2,20,2,3,5,27,2,4,15,3,6,21,24,7,4,27,60,10,6,15,8,21,47,4,13,5,2,32,12,3,22,21,2,3,6,26,18,10,2,43,2,3,33])
        text_length = torch.LongTensor([text.size(0)])

        feats = torch.randn(5, self.model.generator.posterior_encoder.input_conv.in_channels) \
            if self.use_teacher_forcing else None
        feats_length = torch.LongTensor([5]) if feats is not None else None

        sids = torch.LongTensor([0]) \
            if self.model.generator.spks is not None else None

        spembs = torch.randn(self.model.generator.spk_embed_dim) \
            if self.model.generator.spks is not None else None

        lids = torch.LongTensor([0]) \
            if self.model.generator.langs is not None else None

        duration = torch.randn(text.size(0)) \
            if self.predict_duration else None

        return (text, text_length, feats, feats_length, sids, spembs, lids, duration)

    def get_input_names(self):
        return ['text', 'text_length', 'feats', 'feats_length', 'sids', 'spembs', 'lids', 'duration']

    def get_output_names(self):
        return ['wav', 'att_w', 'dur', 'y_length']

    def get_dynamic_axes(self):
        return {
            'text': {0: 'text_length'},
            'feats': {0: 'feats_length'},
            'wav': {0: 'wav_length'},
            'att_w': {0: 'att_w_feat_length',
                      1: 'att_w_text_length'},
            'dur': {0: 'dur_length'}
        }

    def get_model_config(self, path):
        return {
            'model_type': 'VITS',
            'model_path': str(path / 'tts_model.onnx')
        }
