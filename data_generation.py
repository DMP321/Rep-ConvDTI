from rdkit import Chem
import json
import pickle
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict
import numpy as np
import argparse
import random
from sklearn.cluster import KMeans

def deal_kiba(path):
    drugs = []
    prots = []
    
    ligands = json.load(open(path + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(path + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(path + "Y","rb"), encoding='latin1')

    for d in ligands.keys():
        drugs.append(d)
    for t in proteins.keys():
        prots.append(t) 
    data = []
    for row in tqdm(range(affinity.shape[0])):
        lg = ligands[drugs[row]]
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(lg),canonical=True)
        for column in range(affinity.shape[1]):
            if str(affinity[row][column]) == "nan":
                continue
            else:
                if affinity[row][column]>=12.1:
                    label=1.0
                else:
                    label = 0.0
                data.append([prots[column], drugs[row], proteins[prots[column]], lg, label])
    
    data_list = pd.DataFrame(data, columns=["protein_id", "drug_id", "AAS", "SMILE", "Label"])        
    data_list.to_csv(f'./dataset/kiba.csv', index = False)

def deal_davis(path):
    drugs = []
    prots = []
    ligands = json.load(open(path + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(path + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(path + "Y","rb"), encoding='latin1')
    affinity = -np.log10(np.array(affinity)/1e9)
    for d in ligands.keys():
        drugs.append(d)
    for t in proteins.keys():
        prots.append(t)

    data = []

    for row in tqdm(range(len(affinity))):
        lg = ligands[drugs[row]]
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(lg),canonical=True)
        for column in range(affinity[row].shape[0]):
            label = affinity[row][column]
            if label == 5.0:
                data.append([prots[column], drugs[row], proteins[prots[column]], lg, 0.0])
            else:
                data.append([prots[column], drugs[row], proteins[prots[column]], lg, 1.0])
                
    data_list = pd.DataFrame(data, columns=["protein_id", "drug_id", "AAS", "SMILE", "Label"])        
    data_list.to_csv(f'./dataset/davis.csv', index = False)

def get_negative_samples(mask,drug_dissimmat):    
    pos_num = np.sum(mask)     # 193
    pos_id = np.where(mask==1) # 2D postion of mask,2 * 193
    drug_id = pos_id[0]        # 193
    t_id = pos_id[1]           # 193 
    neg_mask = np.zeros_like(mask)  
    for i in range(pos_num):   # for each positive sample  
        d = drug_id[i]
        t = t_id[i] 
        pos_drug = drug_dissimmat[d]  # 10 
        for j in range(len(pos_drug)):
            neg_mask[int(pos_drug[j])][t] = 1 
    return neg_mask 
def get_drug_dissimmat(drug_affinity_matrix,topk):
    drug_num  = drug_affinity_matrix.shape[0]
    drug_dissim_mat = np.zeros((drug_num,topk))
    index_list = np.arange(drug_num)
    for i in range(drug_num):
        score = drug_affinity_matrix[i]
        index_score = list(zip(index_list,score))
        sorted_result = sorted(index_score,key=lambda x: x[1],reverse=False)[1:topk+1]
        drug_id_list = np.zeros(topk)
        for j in range(topk):
            drug_id_list[j] = sorted_result[j][0]            
        drug_dissim_mat[i] = drug_id_list
    return drug_dissim_mat 

    
def method_random(data):
    positive_list = []
    negtive_list = []

    protein_list = list(data.protein_id.unique())
    
    for prot in tqdm(protein_list, total=len(protein_list)):
        dataset = data[data["protein_id"]==prot]
        pos = dataset[dataset["Label"]==1.0].values.tolist()
        neg = dataset[dataset["Label"]==0.0].values.tolist()
        positive_list.extend(pos)
        if len(neg)>len(pos):
            negtive_list.extend(random.sample(neg, len(pos)))
        else:
            negtive_list.extend(neg)

    temp = negtive_list+positive_list
    data = []
    for i in temp:
        if i[-1]==1.0:
            data.append([i[0], i[1], i[2], i[3], 1.0])
        else:
            data.append([i[0], i[1], i[2], i[3], 0.0])
    return data

def method_distance(data, smile_dict):
    protein_list = data["protein_id"].unique().tolist()
    positive_list = []
    negtive_list = []
    
    for prot in tqdm(protein_list, total=len(protein_list)):
        dataset = data[data["protein_id"]==prot]
        pos = dataset[dataset["Label"]==1.0].values.tolist()
        neg = dataset[dataset["Label"]==0.0].values.tolist()
        if len(pos) != 0:
            positive_list.extend(pos)
        else:
            continue
        if len(neg)>len(pos):
            smile = [i[-2] for i in neg]
            representation = [smile_dict[i] for i in smile]
            kmeans = KMeans(n_clusters=len(pos), random_state=1)
            kmeans.fit(representation)
            centers = kmeans.cluster_centers_
            idex = []
            for i in representation:
                distance = np.linalg.norm(np.array(centers) - i.reshape(1,-1), ord=2, axis=1)
                idex.append(np.argsort(distance)[0])
            idex = list(set(idex))
            negtive_list.extend([neg[i] for i in idex])
        else:
            negtive_list.extend(neg)
    temp = negtive_list+positive_list
    data = []
    for i in temp:
        if i[-1]==1.0:
            data.append([i[0], i[1], i[2], i[3], 1.0])
        else:
            data.append([i[0], i[1], i[2], i[3], 0.0])
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generatation dataset')
    parser.add_argument('--dataset', type=str, help='chose dataset')
    parser.add_argument('--method', choices=["random", "distance", "MULGA"], default="none", help='Methods of processing data sets')
    parser.add_argument('--addition', type=str, default="none", help='Additional information is required when selecting other methods of adoption')
    args = parser.parse_args()
    
    
    if(args.dataset == "DAVIS"):
        deal_davis("./dataset/DAVIS/")
        dataset = pd.read_csv("./davis.csv")
    elif(args.dataset == "KIBA"):
        deal_kiba("./dataset/KIBA/")
        dataset = pd.read_csv("./kiba.csv")
    elif(args.dataset == None):
        print("Please input dataset path!")
    else:
        dataset = pd.read_csv(f"{args.dataset}")
    
    if(args.method == "random"):
        data = method_random(dataset)
    elif(args.method == "distance"):
        if args.dataset == "DAVIS":
            with open("./dataset/DAVIS/smile_dict.pickle", "rb") as f:
                smile_dict = pickle.load(f)
        elif args.dataset == "KIBA":
            with open("./dataset/KIBA/smile_dict.pickle", "rb") as f:
                smile_dict = pickle.load(f)
        else:
            smile_dict = pickle.load(args.addition)
        data = method_distance(dataset, smile_dict)
    elif(args.method == "MULGA"):
        if args.dataset == "DAVIS":
            drugs = []
            prots = []
            datasets = './dataset/DAVIS'
            fpath = datasets + '/'
            ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
            proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
            affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
            drug_similar = np.genfromtxt(fpath + "drug-drug_similarities_2D.txt", delimiter=' ')

            for d in ligands.keys():
                lg = ligands[d]
                lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),canonical=True)
                drugs.append(d)
            for t in proteins.keys():
                prots.append(t) 

            mask = np.where(affinity >= 10000, 0, 1)
            drug_dissimmat = get_drug_dissimmat(drug_similar, 4)
            result = get_negative_samples(mask, drug_dissimmat)
            data = []
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    if mask[i][j] == 1 :
                        data.append([drugs[i], prots[j], proteins[prots[j]], ligands[drugs[i]], 1])
                    if result[i][j] == 1:
                        data.append([drugs[i], prots[j], proteins[prots[j]], ligands[drugs[i]], 0])
            
        elif args.dataset == "KIBA":
            drugs = []
            prots = []
            datasets = './dataset/KIBA'
            fpath = datasets + '/'
            ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
            proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
            affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
            drug_similar = np.genfromtxt(fpath + "kiba_drug_sim.txt", delimiter='\t')[:, :-1]

            for d in ligands.keys():
                    lg = ligands[d]
                    lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),canonical=True)
                    drugs.append(d)
            for t in proteins.keys():
                prots.append(t) 

            mask = np.where(np.nan_to_num(affinity) <= 12.1, 0, 1)
            drug_dissimmat = get_drug_dissimmat(drug_similar, 27)
            result = get_negative_samples(mask, drug_dissimmat)
            data = []

            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    if mask[i][j] == 1 :
                        data.append([drugs[i], prots[j], proteins[prots[j]], ligands[drugs[i]], 1])
                    if result[i][j] == 1:
                        data.append([drugs[i], prots[j], proteins[prots[j]], ligands[drugs[i]], 0])
        
    
    data_list = pd.DataFrame(data, columns=["protein_id", "drug_id", "AAS", "SMILE", "Label"])        
    data_list.to_csv(f'./dataset/{args.dataset}_{args.method}.csv', index = False)
        
    
    
    
    

