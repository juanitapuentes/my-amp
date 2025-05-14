import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import argparse
import glob
from tqdm import tqdm
from args import get_args
from dataset import AmpDataset, AmpDatasetWithImages
from models import SequenceTransformer, MultiModalClassifier
# Amino acid vocabulary for tokenization
AA_LIST = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
VOCAB = {aa: i+1 for i, aa in enumerate(AA_LIST)}
VOCAB['PAD'] = 0

def seq_to_ids(seq: str, max_len: int) -> torch.LongTensor:
    ids = [VOCAB.get(c, 0) for c in seq[:max_len]]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, mode: str):
    """
    Run model on data loader, collect probabilities.
    """
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for seq_ids, dist_map, labels in tqdm(loader, desc="Evaluating"):
            seq_ids = seq_ids.to(device)
            dist_map = dist_map.to(device)
            labels   = labels.to(device)
            if mode == 'sequence':
                out = model(seq_ids)
            elif mode == 'distance':
                out = model(dist_map)
            else:
                out = model(seq_ids, dist_map)
            probs = torch.sigmoid(out).cpu().numpy()
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)

    y_probs = np.vstack(all_probs)
    y_true = np.vstack(all_labels)
    return y_true, y_probs


def ensamble(model, test_loader, device, args):
    """
    Ensemble two models by averaging their predictions.
    """

    model1 = copy.deepcopy(model).to(device)
    model2 = copy.deepcopy(model).to(device)

    ckpt1 = torch.load(args.model_fold1, map_location=device)
    ckpt2 = torch.load(args.model_fold2, map_location=device)

    model1.load_state_dict(ckpt1)
    model2.load_state_dict(ckpt2)

    # Evaluate both and average
    y_true1, y_probs1 = evaluate(model1, test_loader, device, args.mode)
    y_true2, y_probs2 = evaluate(model2, test_loader, device, args.mode)
    assert np.array_equal(y_true1, y_true2), "Labels mismatch between folds"
    y_true = y_true1
    y_probs = (y_probs1 + y_probs2) / 2.0
    return y_true, y_probs

def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray):
    """
    Compute per-class AP, F1, and macro AUC/AP/F1.
    """
    num_classes = y_true.shape[1]
    aps = [average_precision_score(y_true[:,i], y_probs[:,i]) for i in range(num_classes)]
    f1s = []
    for i in range(num_classes):
        prec, rec, _ = precision_recall_curve(y_true[:,i], y_probs[:,i])
        f1s.append(np.max(2 * (prec * rec) / (prec + rec + 1e-8)))
    macro_ap = np.mean(aps)
    macro_f1 = np.mean(f1s)
    macro_auc = roc_auc_score(y_true, y_probs, average='macro')
    return aps, f1s, macro_ap, macro_f1, macro_auc


def main():
    args = get_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
 
    test_split = "/home/bcv_researcher/merged_disk2/amp/Database/Test.csv"

    maps_dir = args.maps_dir
    npy_paths = glob.glob(os.path.join(maps_dir, "*.npy"))

    # 2) Initialize running min/max
    global_min = float("inf")
    global_max = -float("inf")

    # 3) Loop once to update
    for p in npy_paths:
        mat = np.load(p)
        # we only care about mat.min() and mat.max()
        m, M = float(mat.min()), float(mat.max())
        if m < global_min: global_min = m
        if M > global_max: global_max = M


    
    # Prepare test dataset and loader
    seq_transform = lambda s: seq_to_ids(s, get_args().seq_max_len)
    test_ds =   AmpDatasetWithImages(
                csv_file   = args.data_csv,
                maps_dir   = args.maps_dir,
                seq_transform = seq_transform,
                split_file = test_split,
                global_min = global_min,
                global_max = global_max,
                img_size   = 224,
                args       = args,           # now accepted
            )
    
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Instantiate two models of given mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.mode=='sequence':
        model = SequenceTransformer(
            vocab_size=len(VOCAB), max_len=args.seq_max_len,
            d_model=args.seq_d_model, n_heads=args.seq_n_heads,
            n_layers=args.seq_n_layers, args=args
        )
    elif args.mode=='distance':
        model = DistanceTransformer(
            max_len=args.dist_max_len, d_model=args.dist_d_model,
            n_heads=args.dist_n_heads, n_layers=args.dist_n_layers,
            args=args
        )
    elif args.mode == 'cross_juanis':


        model = MultiModalClassifier(
            seq_d_model    = args.seq_d_model,
            struct_d_model = 192,                     # was your old vit_out_dim
            n_heads        = args.seq_n_heads,
            num_layers     = args.seq_n_layers,
            num_classes    = args.num_classes,
            vocab_size     = len(VOCAB),
            max_len_seq    = args.seq_max_len,
            img_size       = 224,      # e.g. 224
            patch_size     = 16,          # e.g. 16
            img_channels   = 1,  
        )

    

    else:
        seq_m = SequenceTransformer(
            vocab_size=len(VOCAB), max_len=args.seq_max_len,
            d_model=args.seq_d_model, n_heads=args.seq_n_heads,
            n_layers=args.seq_n_layers, args=args
        )
        dist_m = DistanceTransformer(
            max_len=args.dist_max_len, d_model=args.dist_d_model,
            n_heads=args.dist_n_heads, n_layers=args.dist_n_layers,
            args=args
        )
        model = CrossAttentionModel(
            seq_model=seq_m, dist_model=dist_m,
            d_model=args.seq_d_model, n_heads=args.seq_n_heads,
            num_classes=args.num_classes
        )

    y_true, y_probs = ensamble(model, test_loader, device, args)

    # Compute ensemble metrics
    aps, f1s, mAP, mF1, mAUC = compute_metrics(y_true, y_probs)

    # Print results
    print("=== Ensemble Evaluation Results ===")
    for i, lbl in enumerate(test_ds.LABEL_COLUMNS):
        print(f"{lbl}: AP={aps[i]:.4f}, F1={f1s[i]:.4f}")
    print(f"Mean AP: {mAP:.4f}, Mean F1: {mF1:.4f}, Macro AUC: {mAUC:.4f}")

if __name__=='__main__':
    main()