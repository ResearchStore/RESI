import os
from torch.utils.data import Dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from rdkit import Chem
from tqdm import tqdm
import numpy as np
import json
import torch.nn.functional as F
from sjh_models.MolGNN import Multimodal_KM_1117


def default_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

def get_sample_multiviews(mol_id,graph_dir,img_dir):
    graph = torch.load(os.path.join(graph_dir,mol_id+'.pt'))
    img_2d_path = os.path.join(img_dir, mol_id, f"{mol_id}_0.png")
    img_2d = Image.open(img_2d_path).convert('RGB')
    transform = default_transform()
    img_2d = transform(img_2d)

    img_3d_list = []
    for view_idx in range(1, 7):
        img_3d_path = os.path.join(img_dir, mol_id, f"{mol_id}_{view_idx}.png")
        if os.path.exists(img_3d_path):
            img_3d = Image.open(img_3d_path).convert('RGB')
            img_3d = transform(img_3d)
            img_3d_list.append(img_3d)
        else:
            print(f'视角{view_idx}不存在')
    img_3d_tensor = torch.stack(img_3d_list)
    return {
        'graph': graph,
        'img_2d': img_2d,
        'img_3d': img_3d_tensor,
        'mol_id': mol_id
    }
def getGNN_rep_KM(model,molids_file,graph_save_dir,img_data_dir,device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    reps = []
    model.eval()
    with torch.no_grad():
        for idx, row in tqdm(molids_file.iterrows(), total=len(molids_file), desc="Generating rep_gnn"):
            mol_id = row['molecule ID']
            sample = get_sample_multiviews(mol_id.replace(':', '_'), graph_save_dir, img_data_dir)
            graph = sample['graph'].to(device)
            if graph.edge_index.shape[0] != 2:
                graph.edge_index = torch.zeros((2, 0), dtype=torch.long, device='cuda:0')
                graph.edge_attr = torch.zeros((0, 6), dtype=torch.float32, device='cuda:0')
            img_2d = sample['img_2d'].to(device)
            img_3d = sample['img_3d'].to(device)

            img_2d = img_2d.unsqueeze(0)
            img_3d = img_3d.unsqueeze(0)
            z_graph, z_2d, z_3d, fused_rep ,cat_rep= model.get_GNN_rep(graph, img_2d, img_3d)
            cat_rep = cat_rep.cpu().detach().numpy()
            reps.append(cat_rep[0])
    molids_file['rep_gnn'] = reps
    return molids_file
def update_CVfile_repgnn(CV_data,mol_GNNrep):
    b_mapping = mol_GNNrep[['molecule ID', 'rep_gnn']]
    df_merged = pd.merge(CV_data, b_mapping, on='molecule ID', how='left')
    return df_merged

def update_CVfile_repgnn_FCFP(CV_data,mol_GNNrep):
    b_mapping = mol_GNNrep[['molecule ID', 'rep_gnn']]
    df_merged = pd.merge(CV_data, b_mapping, on='molecule ID', how='left')
    return df_merged
def parse_array(s):
    s = s.strip('[]')
    return np.fromstring(s, sep=' ')
def update_espair_feature_dict(espair_feature_dict,mol_GNNrep):
    for key in espair_feature_dict.keys():
        parts = key.split('_')
        enzyme_id = parts[0]
        molecule_id = '_'.join(parts[1:])
        matches = mol_GNNrep.loc[mol_GNNrep['molecule ID'] == molecule_id, 'rep_gnn']
        if not matches.empty:
            rep_gnn = matches.iloc[0]
        else:
            continue
        old_feature = espair_feature_dict[key]
        esm_feature = old_feature[:1280]
        new_feature = np.concatenate([np.array(esm_feature), rep_gnn])
        if isinstance(espair_feature_dict[key], dict):
            espair_feature_dict[key]["gnn_feature"] = new_feature.tolist()
        else:
            espair_feature_dict[key] = {
                "ecfp_feature": old_feature,
                "gnn_feature": new_feature.tolist()
            }
    keys_to_delete = []
    for key, value in espair_feature_dict.items():
        if not isinstance(value, dict) or "gnn_feature" not in value:
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del espair_feature_dict[key]
    return espair_feature_dict

def run_getGNNRep_KM():

    bestmodel_path = '../save_data/gnn/logs/multimodal_KM/exp_KM_train_1117_20251119_095340/best_model_loss_3.6112.pth'
    weights = torch.load(bestmodel_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Multimodal_KM_1117(
        graph_in_dim=132,
        graph_hidden_dim=512,
        proj_dim=256,
        fusion_out_dim=128
    ).to(device)

    model.load_state_dict(weights)
    molids_file = pd.read_csv('../data/splits/molecule_CVdict.csv')
    img_data_dir = '../data_prepare/ethanol_images6'
    graph_save_dir = '../data_prepare/mol_graphs_v1106'
    mol_GNNrep = getGNN_rep_KM(model, molids_file, graph_save_dir, img_data_dir, device)

    df_train=pd.read_pickle('../data/splits/df_train_CV.pkl')
    df_test=pd.read_pickle('../data/splits/df_test_CV.pkl')
    df_train_gnnrep=update_CVfile_repgnn(df_train,mol_GNNrep)
    df_test_gnnrep = update_CVfile_repgnn(df_test, mol_GNNrep)
    df_train_gnnrep.to_pickle('../data/splits/df_train_gnnrep_CV_1120.pkl')
    df_test_gnnrep.to_pickle('../data/splits/df_test_gnnrep_CV_1120.pkl')
    with open("../data/jsonfiles/espair_feature_dict.json", "r") as f:
        espair_feature_dict = json.load(f)
    espair_feature_gnn_dict=update_espair_feature_dict(espair_feature_dict,mol_GNNrep)
    with open('../data/jsonfiles/espair_feature_gnn_dict_1120.json', 'w') as f:
        json.dump(espair_feature_gnn_dict, f, indent=4)
if __name__ == '__main__':
    run_getGNNRep_KM()