import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from torch_geometric.data import Data, Batch
from PIL import Image
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import torchvision.transforms as T
import random

class PairedEnzymeDataset_Contrast_final_gnnrep_1119(Dataset):
    def __init__(self,  indicestoesid_file, espair_feature_dict, similar_pair_func):

        self.data = indicestoesid_file
        self.espair_feature_dict = espair_feature_dict
        self.similar_pair_func = similar_pair_func

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        enzyme_id = row['Uniprot ID']
        molecule_id = row['molecule ID']
        label = row['Binding']
        input1_ecfp = np.array(self.espair_feature_dict[enzyme_id + '_' + molecule_id]['ecfp_feature'], dtype=np.float32)
        input1_gnn = np.array(self.espair_feature_dict[enzyme_id+'_'+molecule_id]['gnn_feature'], dtype=np.float32)

        query_ecfp_vector = row['ECFP_vector']
        similar_pairs = self.similar_pair_func(enzyme_id, molecule_id, query_ecfp_vector, top_k=10)

        input2_vectors_ecfp = []
        input2_vectors_gnn=[]
        if(len(similar_pairs)!=10):
            print(similar_pairs)
        for (e_sim, s_sim) in similar_pairs:
            if (e_sim+'_'+s_sim) in self.espair_feature_dict:
                vec_ecfp = self.espair_feature_dict[e_sim + '_' + s_sim]['ecfp_feature']
                vec_ecfp = torch.from_numpy(np.array(vec_ecfp, dtype=np.float32)).float()
                vec_gnn = self.espair_feature_dict[e_sim+'_'+s_sim]['gnn_feature']
                vec_gnn = torch.from_numpy(np.array(vec_gnn, dtype=np.float32)).float()
                input2_vectors_ecfp.append(vec_ecfp)
                input2_vectors_gnn.append(vec_gnn)
            else:
                print(e_sim+'_'+s_sim)

        FIXED_LENGTH = 2304
        if len(input2_vectors_ecfp) > 0:
            input2_ecfp = torch.cat(input2_vectors_ecfp, dim=0)
        else:
            input2_ecfp = torch.tensor([], dtype=torch.float32)
        if len(input2_vectors_gnn) > 0:
            input2_gnn = torch.cat(input2_vectors_gnn, dim=0)
        else:
            input2_gnn = torch.tensor([], dtype=torch.float32)
        return {
            'input1_ecfp': torch.from_numpy(input1_ecfp).float(),
            'input1_gnn': torch.from_numpy(input1_gnn).float(),
            'input2_ecfp': input2_ecfp.float(),
            'input2_gnn':input2_gnn.float(),
            'label': torch.tensor(label, dtype=torch.float32)
        }

class GraphDataset_KM_6(Dataset):
    def __init__(self, split_IDs, graph_folder, image_folder,esm1b_dict, target_dict):
        self.all_IDs = split_IDs
        self.graph_folder = graph_folder
        self.image_folder=image_folder
        self.esm1b_dict = esm1b_dict
        self.target_dict = target_dict
        self.transform =  self.default_transform()
    def __len__(self):
        return len(self.all_IDs)
    def default_transform(self):
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
    def __getitem__(self, idx):
        ID = self.all_IDs[idx]

        try:
            parts = ID.split("_")
            if len(parts) == 3:
                uid, cid1, cid2 = parts
                cid = f"{cid1}_{cid2}"
            else:
                uid, cid = parts

            graph_path = os.path.join(self.graph_folder, f"{cid}.pt")
            graph_data = torch.load(graph_path)

            esm1b = torch.tensor(self.esm1b_dict[uid], dtype=torch.float32)

            label = torch.tensor(self.target_dict[ID], dtype=torch.float32)

            img_2d_path = os.path.join(self.image_folder, cid, f"{cid}_0.png")
            img_2d = Image.open(img_2d_path).convert('RGB')
            img_2d = self.transform(img_2d)

            img_3d_list = []
            for view_idx in range(1, 7):
                img_3d_path = os.path.join(self.image_folder, cid, f"{cid}_{view_idx}.png")
                if os.path.exists(img_3d_path):
                    img_3d = Image.open(img_3d_path).convert('RGB')
                    img_3d = self.transform(img_3d)
                    img_3d_list.append(img_3d)
                else:
                    placeholder = torch.zeros(3, 224, 224)
                    img_3d_list.append(placeholder)
                    print(f"Warning: {img_3d_path} not found, using placeholder")

            img_3d_tensor = torch.stack(img_3d_list)

            return graph_data, img_2d, img_3d_tensor, esm1b, label, cid
        except Exception as e:
            print(f"Error loading {ID}: {e}")
            return None
class Pretrain_GNN_Dataset_6(Dataset):
    def __init__(self, root_dir, smiles_file,graph_dir,transform=None, max_atoms=100):

        self.root_dir = root_dir
        self.smiles_file=pd.read_csv(smiles_file)
        self.transform = transform or self.default_transform()
        self.max_atoms = max_atoms
        self.molecule_ids = [filename for filename in os.listdir(root_dir)
                             if os.path.isdir(os.path.join(root_dir, filename))]
        self.graph_dir=graph_dir
    def default_transform(self):
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.molecule_ids)

    def __getitem__(self, idx):
        mol_id = self.molecule_ids[idx]
        mol_id=mol_id.replace(':','_')
        graph = torch.load(os.path.join(self.graph_dir,mol_id+'.pt'))

        img_2d_path = os.path.join(self.root_dir, mol_id, f"{mol_id}_0.png")
        img_2d = Image.open(img_2d_path).convert('RGB')
        img_2d = self.transform(img_2d)

        img_3d_list = []
        for view_idx in range(1, 7):
            img_3d_path = os.path.join(self.root_dir, mol_id, f"{mol_id}_{view_idx}.png")
            if os.path.exists(img_3d_path):
                img_3d = Image.open(img_3d_path).convert('RGB')
                img_3d = self.transform(img_3d)
                img_3d_list.append(img_3d)
            else:
                placeholder = torch.zeros(3, 224, 224)
                img_3d_list.append(placeholder)
                print(f"Warning: {img_3d_path} not found, using placeholder")

        img_3d_tensor = torch.stack(img_3d_list)

        return graph, img_2d, img_3d_tensor