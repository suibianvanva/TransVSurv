import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Dict


class GLU(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.glu = nn.GLU()

    def forward(self, x):
        a = self.fc1(x)
        b = self.fc2(x)
        return self.glu(torch.cat([a, b], dim=-1))


class GateAddNorm(nn.Module):
    def __init__(self, input_size, output_size):
        super(GateAddNorm, self).__init__()
        self.glu = GLU(input_size, input_size)
        self.projection = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x, skip_a):
        x = self.glu(x)
        x = x + skip_a
        return self.layer_norm(self.projection(x))


class GRN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None, context_size=None, dropout=0.0):
        super(GRN, self).__init__()
        hidden_size = input_size if hidden_size is None else hidden_size
        self.lin_a = nn.Linear(input_size, hidden_size)
        self.lin_c = nn.Linear(context_size, hidden_size) if context_size is not None else None
        self.lin_i = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.project_a = nn.Linear(input_size, hidden_size) if hidden_size != input_size else nn.Identity()
        self.gate = GateAddNorm(hidden_size, output_size)

    def forward(self, a: Tensor, c: Optional[Tensor] = None):
        x = self.lin_a(a)
        if c is not None and self.lin_c is not None:
            x = x + self.lin_c(c).unsqueeze(1)
        x = F.elu(x)
        x = self.lin_i(x)
        x = self.dropout(x)
        return self.gate(x, self.project_a(a))


class VariableSelectionNetwork(nn.Module):
    def __init__(self, d_model, variable_num, dropout=0.0):
        super(VariableSelectionNetwork, self).__init__()
        self.joint_grn = GRN(d_model * variable_num,
                             variable_num,
                             hidden_size=d_model,
                             context_size=d_model,
                             dropout=dropout
                             )
        self.variable_grns = nn.ModuleList(
            [GRN(d_model, d_model,
                 dropout=dropout) for _ in range(variable_num)]
        )

    def forward(self, x: Tensor, context: Optional[Tensor] = None,
                missing_mask: Optional[Tensor] = None):


        x_flattened = torch.flatten(x, start_dim=-2)
        logits = self.joint_grn(x_flattened, context)

        if missing_mask is not None:

            logits = logits.masked_fill(missing_mask, float('-inf'))

            all_missing = missing_mask.all(dim=-1, keepdim=True)
            if all_missing.any():
                logits = torch.where(all_missing, torch.zeros_like(logits), logits)

        variable_selection_weights = F.softmax(logits, dim=-1)


        processed_vars = []
        for i, grn in enumerate(self.variable_grns):

            var_data = x[..., i, :]
            processed_vars.append(grn(var_data))
        x_processed = torch.stack(processed_vars, dim=-1)


        selection_result = torch.matmul(x_processed, variable_selection_weights.unsqueeze(-1)).squeeze(-1)

        return selection_result, variable_selection_weights


class StaticCovariateEncoder(nn.Module):
    def __init__(self, d_model, static_len, dropout=0.0):
        super(StaticCovariateEncoder, self).__init__()
        self.static_vsn = VariableSelectionNetwork(d_model, static_len) if static_len else None
        self.grn = GRN(d_model, d_model, dropout=dropout)

    def forward(self, static_input, static_missing_mask=None):
        if static_input is not None and self.static_vsn is not None:

            static_features, static_weights = self.static_vsn(static_input, missing_mask=static_missing_mask)

            return self.grn(static_features),static_weights
        return None


class VSL(nn.Module):
    def __init__(self,
                 d_model: int,
                 static_len: int = None,
                 temporal_len: int = None,
                 dropout: float = 0.0,
                 temporal_variable_names: Optional[List[str]] = None,
                 static_variable_names:Optional[List[str]] = None):
        super(VSL, self).__init__()


        self.static_encoder = StaticCovariateEncoder(
            d_model=d_model,
            static_len=static_len,
            dropout=dropout
        ) if static_len else None


        self.temporal_vsn = VariableSelectionNetwork(
            d_model=d_model,
            variable_num=temporal_len,
            dropout=dropout
        )


        self.temporal_variable_names = temporal_variable_names if temporal_variable_names \
            else [f"temporal_var_{i}" for i in range(temporal_len)]

        self.static_variable_names = static_variable_names if static_variable_names \
            else [f"static_var_{i}" for i in range(static_len)] if static_len else None

    def forward(self, temporal_input: Tensor, static_input: Optional[Tensor] = None,
                temporal_missing_mask: Optional[Tensor] = None,
                static_missing_mask: Optional[Tensor] = None):

        static_weights = None
        if self.static_encoder and static_input is not None:
            context, static_weights = self.static_encoder(static_input, static_missing_mask=static_missing_mask)
        else:
            context = None

        temporal_features, temporal_weights = self.temporal_vsn(temporal_input, context,
                                                                missing_mask=temporal_missing_mask)

        return temporal_features, temporal_weights, static_weights


    def get_variable_importance(self,
                                temporal_weights: Tensor,
                                static_weights: Optional[Tensor] = None) -> Dict[str, Dict[str, float]]:
        importance_dict = {
            "temporal": {},
            "static": {}
        }


        mean_temporal_weights = temporal_weights.mean(dim=[0, 1]).cpu()


        importance_dict["temporal"] = {
            name: weight.item()
            for name, weight in zip(self.temporal_variable_names, mean_temporal_weights)
        }


        if static_weights is not None and self.static_variable_names is not None:
            mean_static_weights = static_weights.mean(dim=0).cpu()
            importance_dict["static"] = {
                name: weight.item()
                for name, weight in zip(self.static_variable_names, mean_static_weights)
            }

        return importance_dict
