# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F
import copy
import os
import random
import wandb
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import glob
from args import get_args
from dataset import AmpDataset, AmpDatasetWithImages
from models import SequenceTransformer, MultiModalClassifier

# Amino acid vocabulary and sequence converter
AA_LIST = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
VOCAB = {aa: i+1 for i, aa in enumerate(AA_LIST)}
VOCAB['PAD'] = 0

def seq_to_ids(seq: str, max_len: int) -> torch.LongTensor:
    """
    Convert amino-acid sequence to fixed-length ID tensor.
    """
    ids = [VOCAB.get(c, 0) for c in seq[:max_len]]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def evaluate(model, loader, device, mode, args):
    """
    Evaluate model and compute loss + metrics.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    y_true, y_probs = [], []

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
            loss = criterion(out, labels)
            total_loss += loss.item()
            probs = torch.sigmoid(out).cpu().numpy()
            y_true.append(labels.cpu().numpy())
            y_probs.append(probs)

    y_true = np.vstack(y_true)
    y_probs = np.vstack(y_probs)
    per_ap = [average_precision_score(y_true[:,i], y_probs[:,i]) for i in range(y_true.shape[1])]
    per_f1 = []
    for i in range(y_true.shape[1]):
        prec, rec, _ = precision_recall_curve(y_true[:,i], y_probs[:,i])
        f1_scores = 2*(prec*rec)/(prec+rec+1e-8)
        per_f1.append(np.max(f1_scores))
    macro_ap  = np.mean(per_ap)
    macro_f1  = np.mean(per_f1)
    macro_auc = roc_auc_score(y_true, y_probs, average='macro')

    return total_loss/len(loader), macro_auc, macro_ap, macro_f1, per_ap, per_f1


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

if __name__ == '__main__':
    args = get_args()

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


    wandb.init(
        project=args.project,
        name=args.run_name or f"run_{args.mode}",
        config=vars(args),
        mode="disabled" if args.wandb==False else "online",
    )

    if args.fold == 1:
        train_split = "/home/bcv_researcher/merged_disk2/amp/Database/Fold1.csv"
        val_split = "/home/bcv_researcher/merged_disk2/amp/Database/Fold2.csv"

    else:
        train_split = "/home/bcv_researcher/merged_disk2/amp/Database/Fold2.csv"
        val_split = "/home/bcv_researcher/merged_disk2/amp/Database/Fold1.csv"

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


    
    # Dataset and loaders
    seq_transform = lambda s: seq_to_ids(s, args.seq_max_len)
    ds_train = AmpDatasetWithImages(
                csv_file   = args.data_csv,
                maps_dir   = args.maps_dir,
                seq_transform = seq_transform,
                split_file = train_split,
                global_min = global_min,
                global_max = global_max,
                img_size   = 224,
                args       = args,           # now accepted
            )
    ds_val = AmpDatasetWithImages(
                csv_file   = args.data_csv,
                maps_dir   = args.maps_dir,
                seq_transform = seq_transform,
                split_file = val_split,
                global_min = global_min,
                global_max = global_max,
                img_size   = 224,
                args       = args,           # now accepted
            )
    tr_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size)

    # Model instantiation
    if args.mode == 'sequence':
        model = SequenceTransformer(
            vocab_size=len(VOCAB),
            max_len=args.seq_max_len,
            d_model=args.seq_d_model,
            n_heads=args.seq_n_heads,
            n_layers=args.seq_n_layers,
            args=args
        )
    elif args.mode == 'distance':
        model = DistanceTransformer(
            max_len=args.dist_max_len,
            d_model=args.dist_d_model,
            n_heads=args.dist_n_heads,
            n_layers=args.dist_n_layers,
            args=args
        )

    elif args.mode == 'joint_fusion':
        model = MultiModalJointTransformer(
            seq_d_model=256,
            vit_out_dim=192,  # must match your ViT output dim
            n_heads=args.seq_n_heads,
            num_layers=4,
            num_classes=args.num_classes,
            vocab_size=len(VOCAB),
            max_len_seq=args.seq_max_len
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

    elif args.mode == 'concat_juanis':
        SEQ_D_MODEL   = 256
        VIT_OUT_DIM   = 192
        NUM_LAYERS    = 4
        NUM_CLASSES   = 5
        MAX_LEN_SEQ   = 200

        model = MultiModalClassifierAll(
            seq_d_model=SEQ_D_MODEL,
            vit_out_dim=VIT_OUT_DIM,
            n_heads=args.seq_n_heads,
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            vocab_size=len(VOCAB),
            max_len_seq=MAX_LEN_SEQ
        )
    else:
        seq_m = SequenceTransformer(
            vocab_size=len(VOCAB),
            max_len=args.seq_max_len,
            d_model=args.seq_d_model,
            n_heads=args.seq_n_heads,
            n_layers=args.seq_n_layers
        )
        dist_m = DistanceTransformer(
            max_len=args.dist_max_len,
            d_model=args.dist_d_model,
            n_heads=args.dist_n_heads,
            n_layers=args.dist_n_layers
        )
        model = CrossAttentionModel(
            seq_model=seq_m,
            dist_model=dist_m,
            d_model=args.seq_d_model,
            n_heads=args.seq_n_heads,
            num_classes=args.num_classes
        )

    # Print model summary
    print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Optimizer and scheduler
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    else:
        scheduler = None

    criterion = nn.BCEWithLogitsLoss()

    # Training loop

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        model.train()
        train_loss = 0.0

        for seq_ids, dist_map, labels in tqdm(tr_loader, desc="Training"):
            seq_ids = seq_ids.to(device)
            dist_map = dist_map.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # --------- FOR CROSS_JUANIS W/ MULTITASK EXTENSION ---------
            if args.mode == 'cross_juanis':
                out = model(seq_ids, dist_map)
                loss = criterion(out, labels)
            elif args.mode == 'sequence':
                out = model(seq_ids)
                loss = criterion(out, labels)
            elif args.mode == 'distance':
                out = model(dist_map)
                loss = criterion(out, labels)
            elif args.mode == 'concat_juanis':
                out = model(seq_ids, dist_map)
                loss = criterion(out, labels)
            elif args.mode == 'joint_fusion':
                out = model(seq_ids, dist_map)
                loss = criterion(out, labels)
            else:
                raise ValueError(f"Unknown mode: {args.mode}")

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(tr_loader)

        # --------------------------
        # Validation and Logging
        # --------------------------
        if epoch % 10 == 0:
            print("Validating...")
            val_loss, val_auc, val_ap, val_f1, per_ap, per_f1 = evaluate(model, val_loader, device, args.mode, args)
            wandb.log({
                "Train Loss": train_loss,
                "Val Loss": val_loss,
                "Epoch": epoch
            })
            print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            wandb.log({
                "Train Loss": train_loss,
                "Epoch": epoch
            })
            print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}")

        if scheduler:
            scheduler.step()

        if epoch % args.eval_interval == 0:
            metrics = {
                "Val AUC": val_auc,
                "Val AP": val_ap,
                "Val F1": val_f1
            }
            for i, label in enumerate(AmpDataset.LABEL_COLUMNS):
                metrics[f"AP_{label}"] = per_ap[i]
                metrics[f"F1_{label}"] = per_f1[i]
            wandb.log(metrics)
            print("-- Full Evaluation Metrics --")
            for k, v in metrics.items():
                print(f" {k}: {v:.4f}")

            # Save model checkpoint every 10 epochs
        if epoch % 10 == 0:
            # save it on the mode folder
            if not os.path.exists("outputs"):
                os.makedirs("outputs")
            if not os.path.exists(os.path.join("outputs", args.mode)):
                os.makedirs(os.path.join("outputs", args.mode))
            if not os.path.exists(os.path.join("outputs", args.mode, str(args.fold))):
                os.makedirs(os.path.join("outputs", args.mode, str(args.fold)))
            if not os.path.exists(os.path.join("outputs", args.mode, str(args.fold),str(args.run_name))):
                os.makedirs(os.path.join("outputs", args.mode, str(args.fold),str(args.run_name)))
                
            torch.save(model.state_dict(), os.path.join("outputs", args.mode, str(args.fold),str(args.run_name), f"model_{args.mode}_{args.run_name}_epoch{epoch}.pth"))
            print(f"Model checkpoint saved at epoch {epoch}")

    # Save final model
    torch.save(model.state_dict(), os.path.join("outputs", args.mode, str(args.fold),str(args.run_name), f"modelFINAL_{args.run_name}.pth"))