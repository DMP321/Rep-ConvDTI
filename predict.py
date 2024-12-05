import pandas as pd
import random
import os
from model import Rep_ConvDTI
from dataset import CustomDataSet, collate_fn
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
import timeit
from tensorboardX import SummaryWriter
import numpy as np
import os
import torch
import xgboost as xgb
import torch.nn as nn
import torch.optim as optim
import argparse
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score,precision_recall_curve, auc

from Bio.PDB import *
from Bio import SeqIO
from rdkit import Chem

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Rep_ConvDTI for Target-Drug Bind')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('--use_multi_gpu', type=bool, default=True, help='use multi_gpu or not')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
    parser.add_argument('--reparam', type=bool, default=True, help='re-param or not')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
    parser.add_argument('--task_type', type=str, default='Classification', help='Classification or regression')
    parser.add_argument('--lradj', type=str, default='type1', help='Learning rate adjust')
    parser.add_argument('--prot_len', type=int, default=1000, help='The length of Target')
    parser.add_argument('--drug_len', type=int, default=100, help='The length of Ligand')
    parser.add_argument('--prot_embedding', type=int, default=128, help='The dim of Targeting embedding')
    parser.add_argument('--drug_embedding', type=int, default=128, help='The dim of Ligand embedding')
    parser.add_argument('--kernel_model', type=str, default = 'self', help="Choosing the modol of kernel_size")
    parser.add_argument('--drug_kernel', type=str, default = '[9, 1, 3, 9, 1]', help="The size of kernel")
    parser.add_argument('--prot_kernel', type=str, default = '[13, 1, 5, 13, 1]', help="The size of kernel")
    parser.add_argument('--drug_channel', type=str, default = '[128, 128, 128, 128, 128, 128]', help="The count of channel")
    parser.add_argument('--prot_channel', type=str, default = '[128, 128, 128, 128, 128, 128]', help="The count of channel")
    parser.add_argument('--predict_target', type=str, help="The file path of prediction target, must be .fasta file")
    parser.add_argument('--predict_ligands', type=str, help="The file path of prediction ligand, must be .sdf file")
    parser.add_argument('--save_path', type=str, default="./", help="The file path of saving predicting result")
    
    
    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # if(args.predict_target == None or args.predict_ligand == None):
    #     raise FileNotFoundError("Please choose the target or ligands to predict")

    xgb_params = {
        "booster" : "gbtree", 
        "nthread" : 4,
        "num_feature" : 1024, 
        "seed" : 1234, 
        "objective" : "binary:logistic", 
        "num_class" : 1, 
        "gamma" : 0.1, 
        "max_depth" : 6, 
        "lambda" : 2, 
        "subsample" : 0.8, 
        "colsample_bytree" : 0.8, 
        "min_child_weight" : 2, 
        "eta" : 0.1 
        }


    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True

    plst = list(xgb_params.items())

    model = Rep_ConvDTI(args).cuda()
    if args.use_multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    model.load_state_dict(torch.load("./model/Rep_ConvDTI_baseModel.pth", weights_only=True))
    
    target_path = args.predict_target
    with open(target_path) as f:
        for record in SeqIO.parse(f, "fasta"):
            protein = record.seq
    print(protein)
    
    ligands_path = args.predict_ligands
    mols = Chem.SDMolSupplier(ligands_path)
    smiles_list = []
    for mol in mols:
        smiles_list.append(Chem.MolToSmiles(mol, canonical=True))
    
    predict_dataset = []
    for smile in smiles_list:
        predict_dataset.append(["-", "-", protein, smile, 0])
    
    predict_set = CustomDataSet(predict_dataset)
    predict_dataset_load = DataLoader(predict_set, batch_size=args.batch_size, collate_fn = collate_fn)
    valid_pbar = tqdm(
                enumerate(
                    BackgroundGenerator(predict_dataset_load)),
                total=len(predict_dataset_load))
    model.eval()
    S = []
    X_predict = None
    for valid_i, valid_data in valid_pbar:
        compounds, proteins, labels = valid_data
        compounds = compounds.cuda()
        proteins = proteins.cuda()
        XGBoost_input = model.module.fussion(compounds, proteins)
        X = XGBoost_input.cpu().detach().numpy()
        if X_predict is None:
            X_predict = X
        else:
            X_predict = np.vstack((X_predict, X))
    x_predict = xgb.DMatrix(X_predict)
    xgb_model = xgb.Booster()
    xgb_model.load_model("./model/Rep_ConvDTI_baseModel_XGB")
    S = xgb_model.predict(x_predict)
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(f"{save_path}results.txt", 'a+') as f:
        for i in range(len(S)):
            f.write(f'{protein}\t{smiles_list[i]}\t{S[i]}')
            f.write("\n")
        
            
    
