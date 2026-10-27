import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from agri_data import direct_price_path as m

device = m._device(torch)
for length in (84, 168):
    model = m.build_model(length, torch=torch, nn=nn, F=F).to(device).eval()
    z = torch.zeros(2, 11, length, device=device)
    observed = torch.zeros_like(z); observed[:, 0, :] = 1
    active = torch.zeros(2, 11, dtype=torch.bool, device=device); active[:, 0] = True
    roles = torch.zeros(2, 11, dtype=torch.long, device=device)
    metadata = torch.zeros(2, 11, 4, device=device); calendar = torch.randn(2, 10, 4, device=device)
    base = model(z, observed, active, roles, metadata, calendar)
    assert base.shape == (2, 10) and torch.equal(base, torch.zeros_like(base))
    z[:, 1:, :] = 1000
    assert torch.equal(base, model(z, observed, active, roles, metadata, calendar))

# Linear calendar outputs retain their direct horizon index.
calendar_model = m.build_model(84, torch=torch, nn=nn, F=F).eval()
with torch.no_grad():
    calendar_model.calendar_head.weight.zero_()
    calendar_model.calendar_head.bias.zero_()
    calendar_model.calendar_head.weight[0, 0] = 1.0
z = torch.zeros(1, 11, 84)
observed = torch.zeros_like(z); observed[:, 0, :] = 1
active = torch.zeros(1, 11, dtype=torch.bool); active[:, 0] = True
roles = torch.zeros(1, 11, dtype=torch.long)
metadata = torch.zeros(1, 11, 4); calendar = torch.zeros(1, 10, 4)
calendar[0, :, 0] = torch.arange(10, dtype=torch.float32)
assert torch.equal(calendar_model(z, observed, active, roles, metadata, calendar)[0],
                   torch.arange(10, dtype=torch.float32))

# Padded/missing auxiliary keys remain permutation-invariant under attention.
torch.manual_seed(22)
permutation_model = m.build_model(84, torch=torch, nn=nn, F=F).eval()
with torch.no_grad():
    permutation_model.attention_head.weight.normal_()
    permutation_model.attention_head.bias.normal_()
z = torch.randn(1, 11, 84)
observed = torch.zeros_like(z); observed[:, :2, :] = 1
active = torch.zeros(1, 11, dtype=torch.bool); active[:, :2] = True
roles = torch.arange(11, dtype=torch.long).remainder(4).unsqueeze(0)
metadata = torch.randn(1, 11, 4); metadata[:, :2, :] = 0
z[:, 2:, :] = torch.randn_like(z[:, 2:, :]) * 100
observed[:, 2:, :] = 0; active[:, 2:] = False
calendar = torch.randn(1, 10, 4)
first = permutation_model(z, observed, active, roles, metadata, calendar)
order = torch.tensor([0, 1, 10, 8, 6, 4, 2, 9, 7, 5, 3])
permuted = permutation_model(z[:, order], observed[:, order], active[:, order],
                             roles[:, order], metadata[:, order], calendar)
assert torch.allclose(first, permuted, rtol=0, atol=1e-6)

torch.manual_seed(7)
model = m.build_model(84, 'base', torch=torch, nn=nn, F=F).to(device).eval()
optimizer = torch.optim.SGD(model.parameters(), lr=.01)
z = torch.randn(4, 11, 84, device=device); observed = torch.ones_like(z)
active = torch.ones(4, 11, dtype=torch.bool, device=device); roles = torch.zeros(4, 11, dtype=torch.long, device=device)
metadata = torch.randn(4, 11, 4, device=device); calendar = torch.randn(4, 10, 4, device=device)
for _ in range(2):
    optimizer.zero_grad(set_to_none=True)
    (model(z, observed, active, roles, metadata, calendar) - 1).square().sum().backward()
    grad = model.attention.in_proj_weight.grad
    optimizer.step()
assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad).item() > 0

records=[]
for i in range(3):
    cols=[]
    for c in range(11):
        cols.append({'normalized': np.random.default_rng(i+c).normal(size=84).tolist() if c == 0 else [0.]*84,
                     'observedMask': [1]*84 if c == 0 else [0]*84,
                     'active': c == 0, 'role': 0 if c == 0 else 1, 'metadata': [0.]*4})
    records.append({'product':'radish','origin':f'2026-04-0{i+1}',
                    'window':{'p0':100.+i,'representativeSigma':.1,'columns':cols},
                    'calendar':[[0.,1.,0.,1.]]*10,'targets':[0.]*10,'targetMask':[False]*10})

# The OOM path partitions a final partial effective minibatch without changing
# its global n_h weighting or population/minibatch scale.
weighted = []
for i in range(5):
    targets = [0.]*10; mask = [False]*10
    if i in (0, 1): targets[0] = 110. + i; mask[0] = True
    if i in (2, 3, 4): targets[1] = 90. + i; mask[1] = True
    if i in (1, 4): targets[2] = 120. + i; mask[2] = True
    columns=[]
    for c in range(11):
        columns.append({'normalized':[.01*(i+1)]*84 if c == 0 else [0.]*84,
                        'observedMask':[1]*84 if c == 0 else [0]*84,
                        'active':c == 0,'role':0 if c == 0 else 1,'metadata':[0.]*4})
    weighted.append({'origin':f'2023-01-0{i+1}',
                     'window':{'p0':100.+i,'representativeSigma':.1,'columns':columns},
                     'calendar':[[0.,1.,0.,1.]]*10,'targets':targets,'targetMask':mask})
counts=np.asarray([r['targetMask'] for r in weighted],dtype=bool).sum(axis=0).tolist()
weight_model=m.build_model(84,torch=torch,nn=nn,F=F).to(device).eval()
full,_,_=m._batch_loss(weight_model,weighted[:3],device,torch,q_scale=100.,mae_weight=1.,
                       horizon_counts=counts,population=5,effective_batch=3)
partial=0.
for start in (0,2):
    loss,_,_=m._batch_loss(weight_model,weighted[start:min(start+2,3)],device,torch,
                            q_scale=100.,mae_weight=1.,horizon_counts=counts,
                            population=5,effective_batch=3)
    partial += float(loss.detach().cpu())
expected=0.
for h,count in enumerate(counts):
    if count == 0: continue
    for record in weighted[:3]:
        if record['targetMask'][h]:
            actual=record['targets'][h]; persistence=record['window']['p0']
            error=abs(persistence-actual)
            expected += error/actual/count + error/100./count
expected *= 5./3./10.
assert abs(float(full.detach().cpu())-expected) < 1e-6
assert abs(partial-expected) < 1e-6

# A flat validation curve must not let patience truncate fixed-epoch refits;
# a resumed terminal checkpoint must not take an extra optimizer update.
fit_records = [{**weighted[0], 'targets':[100.]*10, 'targetMask':[True]*10}]
original_batch_loss, original_full_loss = m._batch_loss, m._full_loss
try:
    m._batch_loss = lambda model, batch, device, torch, **kwargs: (next(model.parameters()).sum()*0., None, None)
    m._full_loss = lambda *args, **kwargs: 1.0
    with tempfile.TemporaryDirectory() as td:
        original_out = m.OUT
        original_progress_path = m.PROGRESS_PATH
        m.OUT = Path(td)
        m.PROGRESS_PATH = Path(td)/'training-progress.json'
        checkpoint = Path(td)/'fixed-fit.pt'
        try:
            fixed = m.fit_model('radish',84,'U','base',fit_records,fit_records,100.,
                                'native-fixed-31',checkpoint,fixed_epochs=31)
            resumed = m.fit_model('radish',84,'U','base',fit_records,fit_records,100.,
                                  'native-fixed-31',checkpoint,fixed_epochs=31)
            assert len(fixed['history']) == len(resumed['history']) == 31
            assert fixed['optimizerUpdates'] == resumed['optimizerUpdates'] == 31
            assert fixed['terminalReason'] == 'fixed_epoch_complete'
        finally:
            m.OUT = original_out
            m.PROGRESS_PATH = original_progress_path
finally:
    m._batch_loss, m._full_loss = original_batch_loss, original_full_loss

with tempfile.TemporaryDirectory() as td:
    path=Path(td)/'saved.pt'
    torch.save({'schemaVersion':1,'fitId':'native-replay','experiment':m.EXPERIMENT_ID,
                'contextDays':84,'inputType':'U','repair':'base','bestEpoch':1,
                'selectedInputChannels':[],'stateDict':{k:v.detach().cpu() for k,v in model.state_dict().items()}},path)
    reloaded,_=m._load_saved_fit_model(path,expected_fit_id='native-replay',expected_details=[])
    assert m._state_dict_agreement(model,reloaded)['stateAgreement']
    original=m._predict_records(model,records,device,torch)
    replay=m._predict_records(reloaded,records,device,torch)
    result=m._compare_saved_model_replay(original,replay)
    assert result['allRowsAgree'] and result['maxAbsoluteDifference'] == 0.0
print('native shapes, U invariance, attention gradients, saved-state replay, global partial-batch weighting, exact fixed-epoch resume, and all-horizon replay passed')
