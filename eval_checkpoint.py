import os, random
import numpy as np
import torch
from resnet_model import MahjongPolicyResNet

# Load dataset
npz_path = 'ruichang_expert_v1.npz'
if not os.path.exists(npz_path):
    raise FileNotFoundError(f'Dataset not found: {npz_path}')
npz = np.load(npz_path)
S = npz['S']  # shape [N, 14, 30]
A = npz['A']  # shape [N]

# Sample a subset for quick validation
sample_size = 2000
indices = random.sample(range(len(S)), sample_size)
S_sample = torch.from_numpy(S[indices]).float()
A_sample = torch.from_numpy(A[indices]).long()

# Load checkpoint
ckpt_path = os.path.join('checkpoints', 'best_policy.pth')
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
state = torch.load(ckpt_path, map_location='cpu')
model = MahjongPolicyResNet().to('cpu')
model.load_state_dict(state['model_state_dict'])
model.eval()

# Legal mask: channel 13 indicates legal actions (>0)
legal_mask = (S_sample[:, 13] > 0).float()

with torch.no_grad():
    logits = model(S_sample, legal_mask=legal_mask)
    preds = logits.argmax(dim=-1)
    acc = (preds == A_sample).float().mean().item()
    print('Sample validation accuracy on 2k examples:', acc)
