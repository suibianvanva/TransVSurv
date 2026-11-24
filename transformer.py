import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super(PositionalEncoding,self).__init__()

        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()


        self.position_embeddings = nn.Parameter(torch.zeros(1, max_len, d_model))


        self._init_weights()

        self.dropout = nn.Dropout(p=dropout)

    def _init_weights(self):
        position = torch.arange(0, self.position_embeddings.size(1)).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.position_embeddings.size(2), 2).float() *
                             (-math.log(10000.0) / self.position_embeddings.size(2)))


        pe = torch.zeros_like(self.position_embeddings[0])
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)


        with torch.no_grad():
            self.position_embeddings.copy_(pe.unsqueeze(0))


        self.position_embeddings.data += torch.randn_like(self.position_embeddings.data) * 0.02

    def forward(self, x):

        seq_len = x.size(1)


        pos_embeddings = self.position_embeddings[:, :seq_len, :]


        return self.dropout(x + pos_embeddings)


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(RelativePositionalEncoding, self).__init__()


        self.max_len = max_len
        self.d_model = d_model


        self.relative_positions = nn.Parameter(
            torch.zeros(2 * max_len - 1, d_model)
        )

        self.dropout = nn.Dropout(p=dropout)


        self._init_weights()

    def _init_weights(self):
        positions = torch.arange(-(self.max_len - 1), self.max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() *
            (-math.log(10000.0) / self.d_model)
        )


        pe = torch.zeros_like(self.relative_positions)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)


        with torch.no_grad():
            self.relative_positions.copy_(pe)

    def get_relative_positions(self, length):

        ranges = torch.arange(length).unsqueeze(0).repeat(length, 1)
        relative_positions = ranges - ranges.transpose(0, 1)


        relative_positions += self.max_len - 1


        return self.relative_positions[relative_positions]

    def forward(self, x):
        seq_len = x.size(1)


        relative_pos_encoding = self.get_relative_positions(seq_len)


        encoded = x.unsqueeze(2) + relative_pos_encoding.unsqueeze(0)
        encoded = torch.mean(encoded, dim=2)

        return self.dropout(encoded)


class LinearDecoder(nn.Module):
    def __init__(self, lookback_time, d_input, pred_time, n_events):
        super().__init__()
        self.pred_time = pred_time
        self.n_events = n_events


        self.decoders = nn.ModuleList([
            nn.Linear(lookback_time * d_input, pred_time)
            for _ in range(n_events)
        ])

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        outputs = []
        for i in range(self.n_events):
            dec_output = self.decoders[i](x)
            outputs.append(dec_output)

        logits = torch.stack(outputs, dim=-1)


        zero_logit = torch.zeros_like(logits[..., :1])
        logits = torch.cat([logits, zero_logit], dim=-1)

        probs = torch.softmax(logits, dim=-1)


        return probs[..., :self.n_events]


class Transformer  (nn.Module):
    def __init__(self,feature_dim,
                 lookback_time,
                 pred_time,
                 n_events,
                 num_heads=4,
                 num_layers=2,
                 hidden_dim=64,
                 d_model=128,
                 dropout=0.5
                 ):
        super(Transformer,self).__init__()


        self.input_embedding = nn.Linear(feature_dim + 1,d_model)


        self.position_encoding = PositionalEncoding(d_model,lookback_time)


        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                        nhead=num_heads,
                                                        dropout=dropout,
                                                        batch_first=True)

        self.encoder = nn.TransformerEncoder(self.encoder_layer,
                                             num_layers=num_layers)


        self.fc1 = nn.Linear(d_model,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)


        self.linear_decoder = LinearDecoder(lookback_time,hidden_dim,pred_time,n_events)

    def forward(self,features,mask,time_idx):


        time_idx = time_idx.unsqueeze(-1).float()
        features = torch.cat([features,time_idx],dim=-1)


        embedded_features = self.input_embedding(features)


        embedded_features = self.position_encoding(embedded_features)


        encoder_output = self.encoder(embedded_features,src_key_padding_mask = mask)


        x = self.fc1(encoder_output)
        x = F.relu(x)
        x = self.fc2(x)


        output = self.linear_decoder(x)

        return output
