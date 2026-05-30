import sys
import csv
import os
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
from torch import nn
from sjh_models.MolGNN import Multimodal_KM_1111, Multimodal_KM_1117
from utils.dataloader import GraphDataset_KM, GraphDataset_KM_6


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)
df_train = pd.read_pickle('./file_records/KM_train.pkl')
df_test = pd.read_pickle('./file_records/KM_test.pkl')
def create_target_dict_KM(df, target_variable_dict):
    for ind in df.index:
        uid = df["Uniprot ID"][ind]
        cid = df["molecule ID"][ind]
        target_variable_dict[uid + "_" + cid.replace(":", "_")] = df["log10_KM"][ind]
    return(target_variable_dict)
mol_files = list(set(df_train["molecule ID"])) + list(set(df_test["molecule ID"]))
mol_files = list(set(mol_files))

target_variable_dict_KM = {}
target_variable_dict_KM = create_target_dict_KM(df = df_train, target_variable_dict = target_variable_dict_KM)
target_variable_dict_KM = create_target_dict_KM(df = df_test, target_variable_dict = target_variable_dict_KM)
def get_uid_cid_IDs(df):
    ID_list = []
    for ind in df.index:
        uid = df["Uniprot ID"][ind]#Enzyme:train:992
        cid = df["molecule ID"][ind]
        ID_list.append(uid + "_" + cid)#Enzyme:train:992_C01204
    return(ID_list)
train_IDs = get_uid_cid_IDs(df_train)#Enzyme:train:992_C01204
test_IDs = get_uid_cid_IDs(df_test)
print(len(train_IDs), len(test_IDs))
uids_list = list(set(df_train["Uniprot ID"])) + list(set(df_test["Uniprot ID"]))
uids_list = list(set(uids_list))
uid_to_emb = {}
embeddings = np.zeros((0,1280))
for uid in uids_list:
    try:
        emb = np.reshape(np.array(list(df_train["ESM1b"].loc[df_train["Uniprot ID"] == uid])[0]), (1,1280))
    except IndexError:
        try:
            emb = np.reshape(np.array(list(df_test["ESM1b"].loc[df_test["Uniprot ID"] == uid])[0]), (1,1280))
        except IndexError:
            # emb = np.reshape(np.array(list(df_validation["ESM1b"].loc[df_validation["Uniprot ID"] == uid])[0]), (1,1280))#sjh注释掉，不然有错误
            print(f"警告: 在数据集中找不到Uniprot ID: {uid}")
    embeddings = np.concatenate([embeddings, emb])
    uid_to_emb[uid] = emb
def custom_collate_fn(batch):
    graph_data_list = []
    img_2d_list = []
    img_3d_list = []
    esm1b_list = []
    labels_list = []
    ids_list = []

    for item in batch:
        graph_data, img_2d, img_3d, esm1b, labels, ids = item
        graph_data_list.append(graph_data)
        img_2d_list.append(img_2d)
        img_3d_list.append(img_3d)
        esm1b_list.append(esm1b)
        labels_list.append(labels)
        ids_list.append(ids)

    graph_batch = Batch.from_data_list(graph_data_list)

    img_2d_batch = torch.stack(img_2d_list)
    img_3d_batch = torch.stack(img_3d_list)
    esm1b_batch = torch.stack(esm1b_list)
    labels_batch = torch.stack(labels_list)

    return graph_batch, img_2d_batch, img_3d_batch, esm1b_batch, labels_batch, ids_list
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

n = len(train_IDs)
random.seed(1)
random.shuffle(train_IDs)
test_IDs = train_IDs[int(0.8*n):]
train_IDs = train_IDs[:int(0.8*n)]

def multiview_contrastive_loss(z_graph, z_2d, z_3d, mol_ids=None, temperature=0.05):

    z_graph = F.normalize(z_graph, dim=1)
    z_2d = F.normalize(z_2d, dim=1)
    z_3d = F.normalize(z_3d, dim=1)

    loss_graph_2d = info_nce_loss(z_graph, z_2d, mol_ids, temperature)
    loss_graph_3d = info_nce_loss(z_graph, z_3d, mol_ids, temperature)
    loss_2d_3d = info_nce_loss(z_2d, z_3d, mol_ids, temperature)

    loss = loss_graph_2d + loss_graph_3d + loss_2d_3d
    return loss, loss_graph_2d, loss_graph_3d, loss_2d_3d

def info_nce_loss(z1, z2, mol_ids=None, temperature=0.1):

    B = z1.size(0)
    device = z1.device

    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature  # [2B, 2B]

    labels = torch.arange(2 * B, device=device)
    labels = (labels + B) % (2 * B)

    if mol_ids is not None:
        mol_ids = list(mol_ids)
        all_ids = mol_ids + mol_ids
        all_ids = np.array(all_ids)

        same_mol = (all_ids[:, None] == all_ids[None, :])
        mask = torch.ones_like(sim, dtype=torch.bool)  # [2B, 2B]
        mask[same_mol] = False
        for i in range(2 * B):
            pos_sample_idx = labels[i]
            mask[i, pos_sample_idx] = True
        sim = sim.masked_fill(~mask, -1e9)

    loss = F.cross_entropy(sim, labels)
    return loss
def joint_loss(z_graph, z_2d,z_3d,mol_ids,logits, labels, lambda_cls=1.0, lambda_cont=0.3):
    loss_mse = F.mse_loss(logits, labels)
    loss_cont, loss_graph_2d, loss_graph_3d, loss_2d_3d = multiview_contrastive_loss(z_graph,z_2d,z_3d,mol_ids)
    loss=lambda_cls * loss_mse + lambda_cont * loss_cont
    return loss,loss_mse,loss_cont,loss_graph_2d,loss_graph_3d,loss_2d_3d
def train_model(train_IDs,test_IDs,graph_folder, image_folder,esm1b_dict,target_dict,batch_size=32, num_epochs=100, lr=1e-3,exp_name=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = GraphDataset_KM_6(
        split_IDs=train_IDs,
        graph_folder=graph_folder,
        image_folder=image_folder,
        esm1b_dict=esm1b_dict,
        target_dict=target_dict
    )
    test_dataset = GraphDataset_KM_6(
        split_IDs=test_IDs,
        graph_folder=graph_folder,
        image_folder=image_folder,
        esm1b_dict=esm1b_dict,
        target_dict=target_dict
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    model = Multimodal_KM_1117(
        graph_in_dim=132,
        graph_hidden_dim=512,
        # img_out_dim=1024,
        proj_dim=256,
        fusion_out_dim=128
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    criterion = nn.MSELoss()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"../save_data/gnn/logs/multimodal_KM/{exp_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "train_log.csv")
    with open(log_file, mode="w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["epoch", "train_loss", "train_mse","train_contrastive","train_graph_2d","train_graph_3d","train_2d_3d",
                             "val_loss", "val_mse","val_contrastive","val_graph_2d","val_graph_3d","val_2d_3d"])
    # writer_tb = SummaryWriter(log_dir=log_dir)

    best_val_loss = float('inf')
    best_auc=float('-inf')
    best_acc = float('-inf')
    for epoch in range(num_epochs):
        model.train()
        # adjust_learning_rate(epoch,scheduler1,optimizer)
        total_samples=0
        epoch_train_loss = 0.0
        epoch_train_mse_loss = 0.0
        epoch_train_contrastive_loss = 0.0
        epoch_train_loss_graph_2d = 0.0
        epoch_train_loss_graph_3d = 0.0
        epoch_train_loss_2d_3d = 0.0
        accumulation_steps = 2
        optimizer.zero_grad()
        for i,batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            graph_data, img_2d, img_3d, esm1b, labels, molids = batch
            # 移动到设备
            graph_data = graph_data.to(device)
            img_2d = img_2d.to(device)
            img_3d = img_3d.to(device)
            esm1b = esm1b.to(device)
            labels = labels.to(device)

            z_graph, z_2d, z_3d, fused_rep, outputs = model(graph_data, img_2d, img_3d, esm1b)
            loss,loss_mse,contrastive_loss, loss_graph_2d, loss_graph_3d, loss_2d_3d = joint_loss(z_graph, z_2d, z_3d,molids,outputs.squeeze(), labels)

            loss = loss / accumulation_steps
            loss_mse = loss_mse / accumulation_steps
            contrastive_loss = contrastive_loss / accumulation_steps
            loss_graph_2d = loss_graph_2d / accumulation_steps
            loss_graph_3d = loss_graph_3d / accumulation_steps
            loss_2d_3d = loss_2d_3d / accumulation_steps
            loss.backward()

            batch_size = labels.size(0)
            epoch_train_loss += loss.item() * accumulation_steps * batch_size
            epoch_train_mse_loss += loss_mse.item() * accumulation_steps * batch_size
            epoch_train_contrastive_loss += contrastive_loss.item() * accumulation_steps * batch_size
            epoch_train_loss_graph_2d += loss_graph_2d.item() * accumulation_steps * batch_size
            epoch_train_loss_graph_3d += loss_graph_3d.item() * accumulation_steps * batch_size
            epoch_train_loss_2d_3d += loss_2d_3d.item() * accumulation_steps * batch_size
            total_samples += batch_size
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        avg_train_loss = epoch_train_loss / total_samples
        avg_train_loss_mse=epoch_train_mse_loss/total_samples
        avg_train_loss_contrastive=epoch_train_contrastive_loss/total_samples
        avg_train_loss_graph_2d=epoch_train_loss_graph_2d/total_samples
        avg_train_loss_graph_3d = epoch_train_loss_graph_3d / total_samples
        avg_train_loss_2d_3d = epoch_train_loss_2d_3d / total_samples

        model.eval()
        epoch_val_loss = 0.0
        epoch_val_loss_mse = 0.0
        epoch_val_loss_contrastive = 0.0
        epoch_val_loss_graph_2d = 0.0
        epoch_val_loss_graph_3d = 0.0
        epoch_val_loss_2d_3d = 0.0
        total_test_samples = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                graph_data, img_2d, img_3d, esm1b, labels, ids = batch
                graph_data = graph_data.to(device)
                img_2d = img_2d.to(device)
                img_3d = img_3d.to(device)
                esm1b = esm1b.to(device)
                labels = labels.to(device)

                z_graph, z_2d, z_3d, fused_rep, outputs = model(graph_data, img_2d, img_3d, esm1b)
                loss, loss_mse, contrastive_loss, loss_graph_2d, loss_graph_3d, loss_2d_3d = joint_loss(z_graph, z_2d,
                                                                                                        z_3d, ids,
                                                                                                        outputs.squeeze(),
                                                                                                        labels)
                batch_size = labels.size(0)
                epoch_val_loss += loss.item() * batch_size
                epoch_val_loss_mse += loss_mse.item() * batch_size
                epoch_val_loss_contrastive += contrastive_loss.item() * batch_size
                epoch_val_loss_graph_2d += loss_graph_2d.item() * batch_size
                epoch_val_loss_graph_3d += loss_graph_3d.item() * batch_size
                epoch_val_loss_2d_3d += loss_2d_3d.item() * batch_size
                total_test_samples += batch_size

        avg_val_loss = epoch_val_loss / total_test_samples
        avg_val_loss_mse=epoch_val_loss_mse/total_test_samples
        avg_val_loss_contrastive=epoch_val_loss_contrastive/total_test_samples
        avg_val_loss_graph_2d=epoch_val_loss_graph_2d/total_test_samples
        avg_val_loss_graph_3d = epoch_val_loss_graph_3d / total_test_samples
        avg_val_loss_2d_3d = epoch_val_loss_2d_3d / total_test_samples
        # ---- Logging ----
        print(f"Epoch {epoch + 1}/{num_epochs} - 训练 Loss: {avg_train_loss:.4f}, "
              f"MSE:{avg_train_loss_mse:.4f}, Contrastive:{avg_train_loss_contrastive:.4f}, "
              f"graph_2d:{avg_train_loss_graph_2d:.4f}, graph_3d:{avg_train_loss_graph_3d:.4f}, 2d_3d:{avg_train_loss_2d_3d:.4f}")
        print(f"Epoch {epoch + 1}/{num_epochs} - 测试集 Loss: {avg_val_loss:.4f}, "
              f"MSE:{avg_val_loss_mse:.4f}, Contrastive:{avg_val_loss_contrastive:.4f}, "
              f"graph_2d:{avg_val_loss_graph_2d:.4f}, graph_3d:{avg_val_loss_graph_3d:.4f}, 2d_3d:{avg_val_loss_2d_3d:.4f}")

        # 写 CSV
        with open(log_file, mode="a", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch+1, avg_train_loss, avg_train_loss_mse, avg_train_loss_contrastive, avg_train_loss_graph_2d, avg_train_loss_graph_3d,avg_train_loss_2d_3d,
                             avg_val_loss, avg_val_loss_mse, avg_val_loss_contrastive, avg_val_loss_graph_2d, avg_val_loss_graph_3d,avg_val_loss_2d_3d])

        if (best_val_loss > avg_val_loss) and (epoch>250):
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(log_dir, f"best_model_loss_{best_val_loss:.4f}.pth")
            torch.save(model.state_dict(), best_model_path)

if __name__ == '__main__':
    train_model(
        train_IDs=train_IDs,
        test_IDs=test_IDs,
        graph_folder='./KM_mol_graphs',
        image_folder='data/pretrain_data/KM_ethanol_images',
        esm1b_dict=uid_to_emb,
        target_dict=target_variable_dict_KM,
        batch_size=128,
        num_epochs=300,
        lr=1e-4,
        exp_name='exp_KM_train')