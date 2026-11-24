import warnings
warnings.filterwarnings('ignore')
import os, sys, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from experiment_config import (
    TEMPORAL_FEATURES, STATIC_FEATURES, N_EVENTS,
    LOOKBACK, PRED_TIME, load_aligned_data,
    fit_temporal_scaler, apply_temporal_scaler, patient_stratified_split,
    patient_stratified_kfold, subset_by_ids,
)
from Data_utils import TimeSeriesDataset
from VSL_transformer import VSL_Transformer

seed = 1234
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HP = dict(num_heads=16, num_layers=2, hidden_dim=256, d_model=512, dropout=0.2071)
LR = 0.000201
WD = 0.0011
BATCH = 256
NUM_WORKERS = int(os.environ.get('TRANSVSURV_WORKERS', '12'))
PATIENCE = 5
MAX_EPOCHS = 100


def make_loader(df, stride, scaler=None, bs=BATCH, shuffle=None):
    if scaler is not None:
        df = apply_temporal_scaler(df, scaler)
    ds = TimeSeriesDataset(df, TEMPORAL_FEATURES, STATIC_FEATURES, 'ID', 'time_idx',
                           LOOKBACK, PRED_TIME, 'event', 'time', 'final_event', stride)
    return DataLoader(ds, batch_size=bs,
                      shuffle=(shuffle if shuffle is not None else stride > 0),
                      collate_fn=TimeSeriesDataset.collate_fn,
                      num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS > 0),
                      pin_memory=(device.type == 'cuda'))


def build_model(use_residual=True):
    return VSL_Transformer(
        temporal_len=len(TEMPORAL_FEATURES), static_len=len(STATIC_FEATURES),
        lookback_time=LOOKBACK, pred_time=PRED_TIME, n_events=N_EVENTS,
        temporal_variable_names=TEMPORAL_FEATURES, static_variable_names=STATIC_FEATURES,
        use_residual=use_residual, **HP).to(device)


def _to_device(batch):
    return [x.to(device) for x in batch]


def pure_nll_loss(y_pred, y_true, eps=1e-7):
    y_pred = torch.clamp(y_pred, min=eps, max=1 - eps)
    h_sum = y_pred.sum(dim=-1)
    survive_prob = torch.clamp(1.0 - h_sum, min=eps)
    valid_mask = (y_true != -1)
    event_mask = (y_true > 0)
    no_event_mask = (y_true == 0) & valid_mask
    event_indices = (y_true - 1).clamp(min=0).long()
    event_hazards = y_pred.gather(-1, event_indices.unsqueeze(-1)).squeeze(-1)
    event_loss = -torch.log(event_hazards + eps)
    no_event_loss = -torch.log(survive_prob + eps)
    loss = torch.where(event_mask, event_loss,
                       torch.where(no_event_mask, no_event_loss, torch.zeros_like(h_sum)))
    return (loss * valid_mask).sum() / (valid_mask.sum() + eps)


def loss_pure_nll(y_pred, y_true, survival_times=None, final_events=None,
                  model=None, l2_lambda=0.0001, **kwargs):
    nll = pure_nll_loss(y_pred, y_true)
    l2 = torch.tensor(0.0, device=y_pred.device)
    if l2_lambda > 0 and model is not None:
        for p in model.parameters():
            if p.requires_grad:
                l2 = l2 + torch.norm(p, 2) ** 2
        l2 = l2_lambda * l2
    total = nll + l2
    return total, {'total': total.item(), 'nll': nll.item(), 'ranking': 0, 'l2': l2.item()}


def train_model(model, train_loader, val_loader, fixed_epochs=None, log_prefix=''):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best_val = float('inf')
    best_state = None
    best_epoch = 0
    patience = 0
    n_epochs = MAX_EPOCHS if fixed_epochs is None else fixed_epochs
    for epoch in range(n_epochs):
        model.train()
        train_ld = None
        for batch in train_loader:
            tf, sf, ev, ti, tg, tm, sm, st, fe, mask = _to_device(batch)
            opt.zero_grad()
            pred, _ = model(tf, sf, mask, ti, temporal_missing_mask=tm, static_missing_mask=sm)
            loss, train_ld = loss_pure_nll(pred, tg, st, fe, model=model)
            loss.backward()
            opt.step()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                tf, sf, ev, ti, tg, tm, sm, st, fe, mask = _to_device(batch)
                pred, _ = model(tf, sf, mask, ti, temporal_missing_mask=tm, static_missing_mask=sm)
                l, _ = loss_pure_nll(pred, tg, st, fe, model=model)
                vl += l.item()
        vl /= max(len(val_loader), 1)
        if train_ld is not None:
            print(f"{log_prefix}Ep{epoch+1}/{n_epochs}: val={vl:.4f} | NLL={train_ld['nll']:.5f} (pure NLL, no ranking)", flush=True)
        else:
            print(f"{log_prefix}Epoch {epoch+1}/{n_epochs}: val_loss={vl:.4f}", flush=True)
        if fixed_epochs is None:
            if vl < best_val:
                best_val = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, best_epoch


def main():
    wt, ks = load_aligned_data()
    print(f"相城: {wt['ID'].nunique()}人 {len(wt)}行; 昆山: {ks['ID'].nunique()}人 {len(ks)}行")

    rest_ids, test_ids = patient_stratified_split(wt, 0.2, seed)
    print(f"内部测试: {len(test_ids)}人, 训练池(80%): {len(rest_ids)}人")
    folds = patient_stratified_kfold(wt, rest_ids, 5, seed)

    best_epochs = []
    for f, (tr_ids, va_ids) in enumerate(folds):
        print(f"\n--- CV Fold {f+1}/5 (train={len(tr_ids)}, val={len(va_ids)}) ---")
        tr = subset_by_ids(wt, tr_ids)
        va = subset_by_ids(wt, va_ids)
        scaler = fit_temporal_scaler(tr)
        trl = make_loader(tr, 1, scaler)
        val = make_loader(va, 1, scaler)
        model = build_model()
        model, bvl, bep = train_model(model, trl, val, log_prefix=f'  F{f+1} ')
        best_epochs.append(bep)
        print(f"  F{f+1} best_val={bvl:.4f} best_epoch={bep}", flush=True)

    retrain_epochs = int(np.median(best_epochs))
    print(f"\nCV完成. 各折best_epoch={best_epochs}, 中位数={retrain_epochs} (用于重训)")

    print("\n=== 在80%训练池上重训最终模型 ===")
    rest = subset_by_ids(wt, rest_ids)
    scaler = fit_temporal_scaler(rest)
    rest_loader = make_loader(rest, 1, scaler)

    monitor_val = make_loader(subset_by_ids(wt, folds[0][1]), 1, scaler)
    final_model = build_model()
    final_model, _, _ = train_model(final_model, rest_loader, monitor_val, fixed_epochs=retrain_epochs,
                                    log_prefix='  retrain ')


if __name__ == '__main__':
    main()
