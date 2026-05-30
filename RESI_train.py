import sys
import csv
from datetime import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from torch.utils.tensorboard import SummaryWriter
from utils.Plot_results import plot_all_metrics
from utils.Sample_functions import BalancedBatchSampler_0925
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
import pandas as pd
from tqdm import tqdm
from utils.dataloader import PairedEnzymeDataset_Contrast_final_gnnrep_1119
from sjh_models.ContrastiveMLP import DoubleMLP_V1212_gnnrep
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, matthews_corrcoef

CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)

def set_seed(seed=42):
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

with open("../data/jsonfiles/train_similar_enzyme_index_CV.json", "r") as f:
    similar_enzyme_dict = json.load(f)
with open("../data/jsonfiles/enzyme_substrates_dict.json", "r") as f:
    enzyme_substrates_dict = json.load(f)
with open("../data/jsonfiles/espair_feature_gnn_dict.json", "r") as f:
    espair_feature_dict = json.load(f)

df_train=pd.read_pickle('../data/splits/df_train_gnnrep_CV.pkl')
df_test=pd.read_pickle('../data/splits/df_test_gnnrep_CV.pkl')

def tanimoto(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.sum(a ** 2)
    norm_b = np.sum(b ** 2)
    denominator = norm_a + norm_b - dot_product
    if denominator == 0:
        return 0.0
    return dot_product / denominator


def generate_similar_pairs(query_enzyme, query_molecule_id, query_ecfp_vector, top_k=10):
    result_pairs = []
    similar_enzymes = similar_enzyme_dict.get(query_enzyme, [])[:top_k]
    for e_sim in similar_enzymes:
        if e_sim not in enzyme_substrates_dict:
            print(f'候选酶{e_sim}不在酶底对关系表')
            continue
        candidate_substrates = enzyme_substrates_dict[e_sim]
        best_s, best_score = None, -1
        for s_id, s_vec in candidate_substrates:
            score = tanimoto(query_ecfp_vector, np.array(s_vec,dtype=np.float32))
            if score > best_score:
                best_score = score
                best_s = s_id
        if best_s is not None:
            result_pairs.append((e_sim, best_s))

    return result_pairs


def supervised_contrastive_loss(z1, z2_all, labels, temperature=0.05):
    batch_size, num_sim, embed_dim = z2_all.shape
    pos_sims = torch.cosine_similarity(
        z1.unsqueeze(1).expand(-1, num_sim, -1),
        z2_all,
        dim=-1
    ) / temperature
    sim_matrix = torch.mm(z1, z1.t()) / temperature
    mask = torch.eye(batch_size, device=z1.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, -1e9)

    losses = []
    for i in range(batch_size):
        if labels[i] == 1:
            positives = pos_sims[i]
            negatives = sim_matrix[i][labels == 0]
        else:
            positives = sim_matrix[i][(labels == 0) & (torch.arange(batch_size, device=z1.device) != i)]
            negatives = torch.cat([
                pos_sims[i],
                sim_matrix[i][labels == 1]
            ], dim=0)

        if positives.numel() == 0:
            continue
        if negatives.numel() == 0:
            continue

        numerator = torch.exp(positives).sum()
        denominator = numerator + torch.exp(negatives).sum()
        loss_i = -torch.log(numerator / denominator)
        losses.append(loss_i)

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=z1.device)
def custom_contrastive_loss_1224(z1, z2_all, labels, temperature=0.05):

    batch_size, num_similar, embed_dim = z2_all.shape

    pos_sims = torch.cosine_similarity(
        z1.unsqueeze(1).expand(-1, num_similar, -1),
        z2_all,
        dim=-1
    ) / temperature
    neg_mask = (labels == 0)

    if not torch.any(neg_mask):
        neg_sims = torch.zeros(batch_size, 1, device=z1.device) - 1e4
    else:
        neg_samples = z1[neg_mask]
        neg_sims = torch.mm(z1, neg_samples.t()) / temperature

    losses = []
    for i in range(batch_size):
        pos_i = pos_sims[i]
        if neg_mask.any():
            neg_i = neg_sims[i]
        else:
            neg_i = neg_sims[i]
        numerator = torch.exp(pos_i).sum()
        denominator = torch.exp(pos_i).sum() + torch.exp(neg_i).sum()
        denominator = torch.clamp(denominator, min=1e-8)
        loss_i = -torch.log(numerator / denominator)
        losses.append(loss_i)
    return torch.stack(losses).mean()
def false_negative_penalty(logits, labels, threshold=0.2):
    probs = F.softmax(logits, dim=1)[:, 1]
    mask = (labels == 1) & (probs < threshold)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return ((threshold - probs[mask]) ** 2).mean()

def joint_loss(z1, z2_all, logits, labels, lambda_cls=1.0, lambda_cont=0.5):
    loss_cls = F.cross_entropy(logits, labels.long(),label_smoothing=0.15)
    loss_cont = supervised_contrastive_loss(z1, z2_all, labels)
    return lambda_cls * loss_cls + lambda_cont * loss_cont

def collate_fn(batch):
    input1_ecfp = torch.stack([item['input1_ecfp'] for item in batch], dim=0)
    input1_gnn = torch.stack([item['input1_gnn'] for item in batch], dim=0)
    input2_ecfp = torch.stack([item['input2_ecfp'] for item in batch], dim=0)
    input2_gnn = torch.stack([item['input2_gnn'] for item in batch], dim=0)
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    return {
        "input1_ecfp": input1_ecfp,
        'input1_gnn': input1_gnn,
        "input2_ecfp": input2_ecfp,
        "input2_gnn":input2_gnn,
        "label": labels
    }
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)


def train_and_validate_fold(
        df_train,df_test,
        train_espair_feature_dict, test_espair_feature_dict,
        similar_pair_func,
        input_dim=2304, embed_dim=512, dropout=0.3,
        lr=1e-3, batch_size=16, num_workers=8,epochs=30,
        exp_name='exp1'
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = PairedEnzymeDataset_Contrast_final_gnnrep_1119(df_train, train_espair_feature_dict,similar_pair_func)
    test_dataset = PairedEnzymeDataset_Contrast_final_gnnrep_1119( df_test, test_espair_feature_dict, similar_pair_func)
    train_sampler = BalancedBatchSampler_0925(
        labels=train_dataset.data['Binding'].values,
        batch_size=batch_size,
        pos_fraction=0.35,
        drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    model = DoubleMLP_V1212_gnnrep(input_dim=input_dim, embed_dim=embed_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=0.01)
    scheduler1 = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1, verbose=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"save_data/logs/{exp_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "train_log.csv")
    with open(log_file, mode="w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["epoch", "train_loss", "train_acc", "train_auc", "train_f1", "train_precision","train_mcc",
                         "val_loss", "val_acc", "val_auc", "val_f1", "val_precision","val_mcc",])

    writer_tb = SummaryWriter(log_dir=log_dir)

    best_val_loss = float('inf')
    best_auc=float('-inf')
    best_acc = float('-inf')
    for epoch in range(epochs):
        model.train()
        # adjust_learning_rate(epoch,scheduler1,optimizer)
        total_samples = 0
        epoch_train_loss = 0.0
        train_all_preds = []
        train_all_probs = []
        train_all_labels = []
        train_gate_all = []
        train_gate_pos = []
        train_gate_neg = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input1_ecfp, input1_gnn, input2_ecfp, input2_gnn, labels = batch['input1_ecfp'].to(device), batch[
                'input1_gnn'].to(device), batch['input2_ecfp'].to(device), batch['input2_gnn'].to(device), batch[
                'label'].to(device)
            z1, z2, output, gate = model(input1_ecfp, input1_gnn, input2_ecfp, input2_gnn)
            loss = joint_loss(z1, z2, output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            epoch_train_loss += loss.item() * batch_size
            total_samples += batch_size

            probs = torch.softmax(output, dim=1)[:, 1]
            preds = torch.argmax(output, dim=1)
            train_all_preds.extend(preds.detach().cpu().numpy())
            train_all_probs.extend(probs.detach().cpu().numpy())
            train_all_labels.extend(labels.detach().cpu().numpy())
            train_gate_all.append(gate.cpu())
            train_gate_pos.append(gate[labels == 1].cpu())
            train_gate_neg.append(gate[labels == 0].cpu())

        avg_train_loss = epoch_train_loss / total_samples
        train_labels = np.array(train_all_labels)
        train_pred_lables = np.array(train_all_preds)
        train_probs = np.array(train_all_probs)

        acc_train = accuracy_score(train_labels, train_pred_lables)
        auc_train = roc_auc_score(train_labels, train_probs) if len(np.unique(train_labels)) > 1 else 0.5
        f1_train = f1_score(train_labels, train_pred_lables)
        pre_train = precision_score(train_labels, train_pred_lables)
        mcc_train = matthews_corrcoef(train_labels, train_pred_lables) if len(np.unique(train_labels)) > 1 else 0.0

        model.eval()
        test_all_labels, test_all_preds, test_all_probs = [], [], []
        test_gate_all = []
        test_gate_pos = []
        test_gate_neg = []
        val_loss = 0.0
        total_test_samples = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                input1_ecfp, input1_gnn, input2_ecfp, input2_gnn, labels = batch['input1_ecfp'].to(device), batch[
                    'input1_gnn'].to(device), batch['input2_ecfp'].to(device), batch['input2_gnn'].to(device), batch[
                    'label'].to(device)
                z1, z2, output, testgate = model(input1_ecfp, input1_gnn, input2_ecfp, input2_gnn)
                loss = joint_loss(z1, z2, output, labels)

                batch_size = labels.size(0)
                val_loss += loss.item() * batch_size
                total_test_samples += batch_size

                probs = torch.softmax(output, dim=1)[:, 1]
                preds = torch.argmax(output, dim=1)
                test_all_preds.extend(preds.detach().cpu().numpy())
                test_all_probs.extend(probs.detach().cpu().numpy())
                test_all_labels.extend(labels.detach().cpu().numpy())

                test_gate_all.append(testgate.cpu())
                test_gate_pos.append(testgate[labels == 1].cpu())
                test_gate_neg.append(testgate[labels == 0].cpu())
        avg_val_loss = val_loss / total_test_samples
        y_true = np.array(test_all_labels)
        y_pred = np.array(test_all_preds)
        y_probs = np.array(test_all_probs)

        acc_test = accuracy_score(y_true, y_pred)
        f1_test = f1_score(y_true, y_pred)
        auc_test = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
        pre_test = precision_score(y_true, y_pred)
        mcc_test = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0


        print(f"Epoch {epoch + 1}/{epochs} - 训练 Loss: {avg_train_loss:.4f}, "
              f"Acc: {acc_train:.4f}, AUC: {auc_train:.4f}, F1: {f1_train:.4f}, Precision: {pre_train:.4f}, "
              f"MCC:{mcc_train:.4f}")
        print(f"Epoch {epoch + 1}/{epochs} - 测试集 Loss: {avg_val_loss:.4f}, "
              f"Acc: {acc_test:.4f}, AUC: {auc_test:.4f}, F1: {f1_test:.4f}, Precision: {pre_test:.4f} "
              f"MCC:{mcc_test:.4f}")

        with open(log_file, mode="a", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch + 1, avg_train_loss, acc_train, auc_train, f1_train, pre_train, mcc_train,
                                 avg_val_loss, acc_test, auc_test, f1_test, pre_test, mcc_test])
        # 写 TensorBoard
        writer_tb.add_scalar("Loss/train", avg_train_loss, epoch)
        writer_tb.add_scalar("Loss/val", avg_val_loss, epoch)
        writer_tb.add_scalar("Acc/train", acc_train, epoch)
        writer_tb.add_scalar("Acc/val", acc_test, epoch)
        writer_tb.add_scalar("AUC/train", auc_train, epoch)
        writer_tb.add_scalar("AUC/val", auc_test, epoch)
        writer_tb.add_scalar("F1/train", f1_train, epoch)
        writer_tb.add_scalar("F1/val", f1_test, epoch)
        writer_tb.add_scalar("Precision/train", pre_train, epoch)
        writer_tb.add_scalar("Precision/val", pre_test, epoch)
        writer_tb.add_scalar("MCC/train", mcc_train, epoch)
        writer_tb.add_scalar("MCC/val", mcc_test, epoch)
        if (auc_test > best_auc) and (epoch > 100):
            best_auc = auc_test
            best_model_path = os.path.join(log_dir, f"best_model_auc_{best_auc:.4f}.pth")  
            torch.save(model.state_dict(), best_model_path)
        if (acc_test > best_acc) and (epoch > 100):
            best_acc = acc_test
            best_model_path = os.path.join(log_dir, f"best_model_acc_{best_acc:.4f}.pth") 
            torch.save(model.state_dict(), best_model_path)
    writer_tb.close()
    plot_all_metrics(log_file)
if __name__ == '__main__':
    train_and_validate_fold(
        df_train=df_train,
        df_test=df_test,
        train_espair_feature_dict=espair_feature_dict,
        test_espair_feature_dict=espair_feature_dict,
        similar_pair_func=generate_similar_pairs,
        input_dim=1380,
        embed_dim=256,
        dropout=0.6,
        lr=5e-5,
        batch_size=256,
        num_workers=16,
        epochs=400,
        exp_name='exp_1213'
    )