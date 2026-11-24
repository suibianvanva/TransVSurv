import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F

from VSL import VSL
from transformer import LinearDecoder,PositionalEncoding,LearnablePositionalEncoding,RelativePositionalEncoding


class FeatureEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.temporal_embedding = nn.Linear(1, d_model)
        self.static_embedding = nn.Linear(1, d_model)
        self.time_embedding = nn.Linear(1,d_model)


        nn.init.xavier_uniform_(self.temporal_embedding.weight)
        nn.init.xavier_uniform_(self.static_embedding.weight)
        nn.init.xavier_uniform_(self.time_embedding.weight)

    def forward(self, temporal_input, static_input, time_idx):


        temporal_embedded = self.temporal_embedding(temporal_input)
        static_embedded = self.static_embedding(static_input)

        time_idx = time_idx.float()
        time_idx = time_idx.unsqueeze(-1)
        time_embedded = self.time_embedding(time_idx)


        return temporal_embedded, static_embedded, time_embedded


class GatedResidual(nn.Module):
    def __init__(self,hidden_dim):
        super(GatedResidual,self).__init__()


        self.gate = nn.Sequential(nn.Linear(hidden_dim*2,hidden_dim),
                                  nn.LayerNorm(hidden_dim),
                                  nn.Sigmoid()
                                  )
    def forward(self,x,residual):


        gate_input = torch.cat([x,residual],dim=-1)

        gate_weight = self.gate(gate_input)

        return x + gate_weight * residual


class NoResidual(nn.Module):
    def forward(self, x, residual):
        return x


class VSL_Transformer(nn.Module):
    def __init__(self,
                 temporal_len,
                 static_len,
                 lookback_time,
                 pred_time,
                 n_events,
                 num_heads=4,
                 num_layers=2,
                 hidden_dim=64,
                 d_model=128,
                 dropout=0.5,
                 temporal_variable_names=None,
                 static_variable_names=None,
                 use_residual=True
                 ):
        super(VSL_Transformer, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.feature_embedding = FeatureEmbedding(d_model)


        self.vsl = VSL(
            d_model=d_model,
            static_len=static_len,
            temporal_len=temporal_len,
            dropout=dropout,
            temporal_variable_names=temporal_variable_names,
            static_variable_names=static_variable_names
        )

        self.position_encoding = PositionalEncoding(d_model, lookback_time)


        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model,
                                       nhead=num_heads,
                                       dropout=dropout,
                                       batch_first=True)
            for _ in range(num_layers)
        ])


        self.fc1 = nn.Linear(d_model , hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.residual_projection = nn.Linear(d_model, hidden_dim)
        self.gated_residual = GatedResidual(hidden_dim) if use_residual else NoResidual()
        self.linear_decoder = LinearDecoder(lookback_time, hidden_dim, pred_time, n_events)
        self.dropout = nn.Dropout(dropout)


        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.residual_projection.weight)


        self.num_heads = num_heads
        self.num_layers = num_layers


    def forward(self, temporal_features, static_features, mask, time_idx,
                temporal_missing_mask=None, static_missing_mask=None):


        temporal_embedded, static_embedded, time_embedded = self.feature_embedding(
            temporal_features, static_features, time_idx
        )
        temporal_embedded = self.layer_norm(temporal_embedded)


        temporal_features, temporal_weights, static_weights = self.vsl(
            temporal_embedded, static_embedded,
            temporal_missing_mask=temporal_missing_mask,
            static_missing_mask=static_missing_mask
        )


        temporal_features = temporal_features + time_embedded


        temporal_features = temporal_features * (~mask).unsqueeze(-1)


        residual = self.residual_projection(temporal_features)


        temporal_features = self.position_encoding(temporal_features)


        encoder_attentions = []
        x = temporal_features

        for layer in self.encoder_layers:
            with torch.no_grad():
                attn_output, attn_weights = layer.self_attn(
                    x, x, x,
                    attn_mask=None,
                    key_padding_mask=mask,
                    need_weights=True
                )

                encoder_attentions.append(attn_weights)

            x = layer(x,src_key_padding_mask=mask)


        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.dropout(F.gelu(self.fc2(x)))


        x = self.gated_residual(x,residual)


        output = self.linear_decoder(x)

        return output, {
            'temporal_selection_weights': temporal_weights,
            'static_selection_weights': static_weights,
            'encoder_attentions': encoder_attentions,
        }
