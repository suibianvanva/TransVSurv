import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data,
        temporal_features,
        static_features,
        id_col,
        time_idx_col,
        lookback_time,
        pred_time,
        event_col='event',
        survival_time_col=None,
        final_event_col=None,
        stride=1
    ):
        self.data = data
        self.temporal_features = temporal_features
        self.static_features = static_features
        self.id_col = id_col
        self.time_idx_col = time_idx_col
        self.lookback_time = lookback_time
        self.pred_time = pred_time
        self.event_col = event_col
        self.survival_time_col = survival_time_col
        self.final_event_col = final_event_col
        self.stride = stride


        self._pdata = {}


        self.windows = []
        for pid in self.data[self.id_col].unique():
            patient_data = self.data[self.data[self.id_col] == pid]
            n_samples = len(patient_data)
            if n_samples < 2:
                continue


            final_event = 0
            if self.final_event_col is not None:
                final_event = patient_data[self.final_event_col].iloc[0]


            self._pdata[pid] = {
                'temporal': patient_data[self.temporal_features].to_numpy().astype(np.float32),
                'static': patient_data[self.static_features].to_numpy()[0].astype(np.float32),
                'event': patient_data[self.event_col].to_numpy().astype(np.int64),
                'time_idx': patient_data[self.time_idx_col].to_numpy().astype(np.int64),
                'n_samples': n_samples,
                'survival_time': float(patient_data[self.survival_time_col].iloc[0]) if self.survival_time_col is not None else None,
                'final_event': int(final_event) if self.final_event_col is not None else None,
            }

            if stride == 0:

                if final_event > 0:

                    pred_start = n_samples - 1
                else:

                    pred_start = max(1, min(self.lookback_time, n_samples - self.pred_time))
                    if pred_start >= n_samples:
                        pred_start = n_samples - 1
                self.windows.append((pid, pred_start))
            else:

                for pred_start in range(1, n_samples, stride):
                    self.windows.append((pid, pred_start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        patient_id, pred_start = self.windows[idx]
        d = self._pdata[patient_id]
        n_samples = d['n_samples']


        n_real_lookback = min(pred_start, self.lookback_time)
        n_pad = self.lookback_time - n_real_lookback
        s0 = pred_start - n_real_lookback


        temporal_lb = d['temporal'][s0:pred_start]
        temporal_missing = np.isnan(temporal_lb)
        temporal_lb = np.nan_to_num(temporal_lb, nan=0.0)
        if n_pad > 0:
            temporal_lb = np.pad(temporal_lb, ((n_pad, 0), (0, 0)), constant_values=0)
            temporal_missing = np.pad(temporal_missing, ((n_pad, 0), (0, 0)), constant_values=True)
        temporal_tensor = torch.tensor(temporal_lb, dtype=torch.float32).unsqueeze(-1)
        temporal_missing_tensor = torch.tensor(temporal_missing, dtype=torch.bool)


        static_vals = d['static']
        static_missing = np.isnan(static_vals)
        static_vals = np.nan_to_num(static_vals, nan=0.0)
        static_tensor = torch.tensor(static_vals, dtype=torch.float32).unsqueeze(-1)
        static_missing_tensor = torch.tensor(static_missing, dtype=torch.bool)


        event_window = d['event'][s0:pred_start]
        time_idx_window = d['time_idx'][s0:pred_start]
        if n_pad > 0:
            event_window = np.pad(event_window, (n_pad, 0), constant_values=-1)
            time_idx_window = np.pad(time_idx_window, (n_pad, 0), constant_values=0)


        pred_end = min(pred_start + self.pred_time, n_samples)
        target = d['event'][pred_start:pred_end].copy().astype(np.int64)


        if self.survival_time_col is not None:
            survival_time = d['survival_time']
            available_time_idx = d['time_idx'][pred_start:pred_end]
            for i, t_idx in enumerate(available_time_idx):
                if t_idx > survival_time:
                    target[i] = -1


        if len(target) < self.pred_time:
            target = np.pad(target, (0, self.pred_time - len(target)), constant_values=-1)

        result = [
            temporal_tensor,
            static_tensor,
            torch.tensor(event_window, dtype=torch.long),
            torch.tensor(time_idx_window, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            temporal_missing_tensor,
            static_missing_tensor,
        ]
        if self.survival_time_col is not None:
            result.append(torch.tensor(d['survival_time'], dtype=torch.float32))
        if self.final_event_col is not None:
            result.append(torch.tensor(d['final_event'], dtype=torch.float32))
        return tuple(result)

    @staticmethod
    def collate_fn(batch):
        n = len(batch[0])
        if n == 9:

            (temporal_features, static_features, events, time_idxs, targets,
             t_missing, s_missing, survival_times, final_events) = zip(*batch)
            temporal_padded = torch.stack(temporal_features)
            static_padded = torch.stack(static_features)
            events_padded = torch.stack(events)
            time_idxs_padded = torch.stack(time_idxs)
            targets_padded = torch.stack(targets)
            t_missing_padded = torch.stack(t_missing)
            s_missing_padded = torch.stack(s_missing)
            mask = (events_padded == -1)
            time_idxs_padded[time_idxs_padded == -1] = 0
            return (temporal_padded, static_padded, events_padded, time_idxs_padded, targets_padded,
                    t_missing_padded, s_missing_padded,
                    torch.stack(survival_times), torch.stack(final_events), mask)


        has_survival_time = False
        has_final_event = False
        if n == 6:
            if isinstance(batch[0][5], torch.FloatTensor) or isinstance(batch[0][5], torch.Tensor):
                has_survival_time = True
        elif n == 7:
            has_survival_time = True
            has_final_event = True

        if has_survival_time and has_final_event:
            (temporal_features, static_features, events, time_idxs, targets, survival_times, final_events) = zip(*batch)
        elif has_survival_time:
            (temporal_features, static_features, events, time_idxs, targets, survival_times) = zip(*batch)
        elif has_final_event:
            (temporal_features, static_features, events, time_idxs, targets, final_events) = zip(*batch)
        else:
            (temporal_features, static_features, events, time_idxs, targets) = zip(*batch)


        temporal_padded = nn.utils.rnn.pad_sequence(temporal_features, batch_first=True, padding_value=0.0)


        static_padded = torch.stack(static_features)


        events_padded = nn.utils.rnn.pad_sequence(events, batch_first=True, padding_value=-1)
        time_idxs_padded = nn.utils.rnn.pad_sequence(time_idxs, batch_first=True, padding_value=-1)
        targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-1)


        mask = (events_padded == -1)
        time_idxs_padded[time_idxs_padded == -1] = 0

        out = [temporal_padded, static_padded, events_padded, time_idxs_padded, targets_padded]
        if has_survival_time:
            out.append(torch.stack(survival_times))
        if has_final_event:
            out.append(torch.stack(final_events))
        out.append(mask)
        return tuple(out)
