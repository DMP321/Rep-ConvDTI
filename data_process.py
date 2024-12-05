import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
import re
import csv
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
import matplotlib.pyplot as plt



CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25


def reading_fasta(path):
    ids = []
    sequences = []
    with open(path, "r") as file_sequences:
        sequence = ""
        for line in file_sequences:
            if line.startswith(">"):
                single_id = line[1:].split(" ")[0]
                ids.append(single_id)
                if sequence != "":
                    sequences.append(sequence)
                    sequence = ""
            else:
                sequence += line[:-1]
        sequences.append(sequence)
    return sequences

def reading_sdf(path):
    compounds = []
    mols = Chem.SDMolSupplier(path)
    for mol in mols:
        compounds.append(Chem.MolToSmiles(mol, isomericSmiles=True))
    return compounds

def generate_dataset(compounds, target, label):
    
    data = []
    
    for c in compounds:
        data.append((c, target, label))
    
    return data

def K_Fold(datasets, proteins_list, i, k=5, mothod="norm"):
    
    if mothod == "norm":
        fold_size = len(datasets) // k  
        val_start = i * fold_size
        if i != k - 1 and i != 0:
            val_end = (i + 1) * fold_size
            validset = datasets[val_start:val_end]
            trainset = datasets[0:val_start] + datasets[val_end:]
        elif i == 0:
            val_end = fold_size
            validset = datasets[val_start:val_end]
            trainset = datasets[val_end:]
        else:
            validset = datasets[val_start:] 
            trainset = datasets[0:val_start]
    elif mothod == "unseen":
        trainset = []
        validset = []
        fold_size = len(proteins_list) // k
        val_start = i * fold_size
        if i != k - 1 and i != 0:
            val_end = (i + 1) * fold_size
            valid_prot = proteins_list[val_start:val_end]
            train_prot = proteins_list[0:val_start] + proteins_list[val_end:]
        elif i == 0:
            val_end = fold_size
            valid_prot = proteins_list[val_start:val_end]
            train_prot = proteins_list[val_end:]
        else:
            valid_prot = proteins_list[val_start:]
            train_prot = proteins_list[0:val_start]
        
        temp = []
        for prot in valid_prot:
            validset.extend([z for z in datasets if z[0]==prot])
            temp.append(prot)
            if len(validset) > int(len(datasets)*0.2):
                break
        p_train = [z for z in valid_prot if z not in temp]

        if len(p_train) != 0:
            train_prot.extend(p_train)
        for prot in train_prot:
            trainset.extend([z for z in datasets if z[0]==prot])
    
    trainset = pd.DataFrame(trainset, columns=["protein_id", "drug_id", "AAS", "SMILE", "Label"])
    validset = pd.DataFrame(validset, columns=["protein_id", "drug_id", "AAS", "SMILE", "Label"])

    return trainset, validset
