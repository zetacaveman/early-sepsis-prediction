# early-sepsis-prediction

Temporal Convolutional Network (TCN) for ICU Time-Series (PhysioNet 2019 Sepsis)



## Data assumptions

Tabular ICU time series with columns like:

	•	PatientID, Hour (integer hours since admission),
	•	feature columns (vitals/labs),
	•	label column SepsisLabel (0/1).
	•	Variable length per patient (some have 4 hours, others 45+, etc.).
	•	Optional: parquet files (.parquet) for faster I/O.

Note: Many datasets have a bookkeeping row at Hour = 0 that’s mostly empty. We usually drop Hour==0 so sequences start at 1.

## Environment

python >= 3.9
pytorch >= 2.0
pandas
numpy
scikit-learn
pyarrow        # if reading parquet
matplotlib     # optional (plots)




## Rough ideas

Padding + masking: pad sequences to same length per batch.

Normalization: compute mean/std on training set only.

Loss: BCEWithLogitsLoss (raw logits input).

Imbalance: optional class weighting

valuation: AUROC/AUPRC (threshold-free) + confusion matrix.



# Key components

## Load + normalize

```python
def load_dataframe(path):
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    df = df[df['Hour'] > 0].sort_values(['PatientID','Hour']).reset_index(drop=True)
    return df

def normalize_train_apply_val(train_df, val_df=None):
    means = train_df[FEATURE_COLS].mean()
    stds  = train_df[FEATURE_COLS].std().replace(0, 1e-6)
    train_df[FEATURE_COLS] = (train_df[FEATURE_COLS] - means)/stds
    if val_df is not None:
        val_df[FEATURE_COLS] = (val_df[FEATURE_COLS] - means)/stds
    return train_df, val_df, means, stds
```

## Build patient sequences
```python
def df_to_patients(df):
    patients = []
    for pid, g in df.groupby('PatientID'):
        g = g.sort_values('Hour')
        if g.empty: continue
        y = int(g['SepsisLabel'].fillna(0).max())
        X = g[FEATURE_COLS].copy().fillna(0.0)
        X_tensor = torch.from_numpy(X.to_numpy(dtype='float32'))
        patients.append({'patient_id': pid, 'series': X_tensor, 'label': torch.tensor(y).float(), 'length': X_tensor.shape[0]})
    return patients, len(FEATURE_COLS)
```

## Dataset + collate (padding)
```python
class TCNDataset(Dataset):
    def __init__(self, patients): self.patients = patients
    def __len__(self): return len(self.patients)
    def __getitem__(self, i): return self.patients[i]

def collate_tcn(batch):
    lengths = [b['length'] for b in batch]
    max_len = max(lengths)
    feat_dim = batch[0]['series'].shape[1]
    feats = torch.zeros(len(batch), max_len, feat_dim)
    mask  = torch.zeros(len(batch), max_len)
    labels = torch.zeros(len(batch))
    last_idx = torch.zeros(len(batch), dtype=torch.long)

    for i,b in enumerate(batch):
        T = b['length']
        feats[i,:T,:] = b['series']
        mask[i,:T] = 1.0
        labels[i] = b['label']
        last_idx[i] = T-1
    feats = feats.permute(0,2,1)
    return {'x':feats,'mask':mask,'y':labels,'last_idx':last_idx}
```
## TCN model


```python
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation, dropout=0.1):
        super().__init__()
        pad = (k-1)*dilation
        self.c1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=dilation)
        self.c2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=dilation)
        self.r = nn.Conv1d(in_ch, out_ch, 1) if in_ch!=out_ch else nn.Identity()
        self.a, self.d = nn.ReLU(), nn.Dropout(dropout)
    def forward(self,x):
        T = x.size(-1)
        y = self.d(self.a(self.c1(x)))[:,:,:T]
        y = self.d(self.a(self.c2(y)))[:,:,:T]
        return y + self.r(x)[:,:,:T]

class TCNModel(nn.Module):
    def __init__(self,in_ch,hidden=64,levels=4,k=3,dropout=0.1):
        super().__init__()
        layers=[]; ch=in_ch
        for i in range(levels):
            layers.append(TemporalBlock(ch,hidden,k,2**i,dropout))
            ch=hidden
        self.tcn=nn.Sequential(*layers)
        self.head=nn.Conv1d(hidden,1,1)
    def forward(self,x):
        return self.head(self.tcn(x)).squeeze(1)
```
## Training step

```python
def train_step(model,batch,opt,device,loss_fn):
    model.train()
    x,y=batch['x'].to(device),batch['y'].to(device)
    li=batch['last_idx'].to(device)
    logits=model(x); B=logits.size(0)
    logits_last=logits[torch.arange(B,device=device),li]
    loss=loss_fn(logits_last,y)
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p=torch.sigmoid(logits_last); preds=(p>0.5).float()
        acc=(preds==y).float().mean()
    return loss.item(), acc.item()
```
## Evaluation (AUROC, AUPRC, CM, Recall)


```python
@torch.no_grad()
def eval_epoch(model,loader,device,threshold=0.5):
    model.eval(); all_probs,all_labels=[],[]; total_loss=0; n=0
    lf=nn.BCEWithLogitsLoss()
    for b in loader:
        x,y=b['x'].to(device),b['y'].to(device); li=b['last_idx'].to(device)
        l=model(x); B=l.size(0); llast=l[torch.arange(B,device=device),li]
        loss=lf(llast,y); total_loss+=loss.item(); n+=1
        probs=torch.sigmoid(llast).cpu().numpy(); labels=y.cpu().numpy()
        all_probs.extend(probs); all_labels.extend(labels)
    probs=np.array(all_probs); labels=np.array(all_labels)
    from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix,recall_score,precision_score,f1_score
    auroc=roc_auc_score(labels,probs) if len(np.unique(labels))>1 else np.nan
    auprc=average_precision_score(labels,probs) if len(np.unique(labels))>1 else np.nan
    preds=(probs>threshold).astype(int)
    cm=confusion_matrix(labels,preds)
    recall=recall_score(labels,preds,zero_division=0)
    precision=precision_score(labels,preds,zero_division=0)
    f1=f1_score(labels,preds,zero_division=0)
    return {'val_loss':total_loss/max(1,n),'val_auroc':auroc,'val_auprc':auprc,'cm':cm,'recall':recall,'precision':precision,'f1':f1}
```

## Training loop (timed)

```python
for epoch in range(epochs):
    t0=time.perf_counter()
    losses,accs=[],[]
    for b in train_loader:
        loss,acc=train_step(model,b,opt,device,loss_fn)
        losses.append(loss); accs.append(acc)
    dt=time.perf_counter()-t0
    log=f"epoch {epoch:02d} | time={dt:.1f}s | loss={sum(losses)/len(losses):.4f} | acc={sum(accs)/len(accs):.4f}"
    if val_loader is not None:
        m=eval_epoch(model,val_loader,device)
        log+=f" | val_loss={m['val_loss']:.4f} AUROC={m['val_auroc']:.3f} AUPRC={m['val_auprc']:.3f} F1={m['f1']:.3f}"
    print(log)
```


# Notes
	•	NaN loss → fill missing data, check std>0.
	•	Accuracy = 1.0 suddenly → probably only last batch printed.
	•	Imbalance: start unweighted; try pos_weight ≤ 5 if recall too low.
	•	Dropout: increase to ~0.3 if overfitting.

