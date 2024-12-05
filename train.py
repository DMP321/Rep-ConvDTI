import pandas as pd
import random
import os
from data_process import K_Fold
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

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score,precision_recall_curve, auc
from torch.utils.data import random_split
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Rep_ConvDTI for Target-Drug Bind')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('--use_multi_gpu', type=bool, default=True, help='use multi_gpu or not')
    parser.add_argument('--gpu', type=int, default=0, help='gpu_device_id')
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
    parser.add_argument('--reparam', type=bool, default=True, help='re-param or not')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='optimizer learning rate')
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    parser.add_argument('--num_workers', type=int, default=48, help='data loader num workers')
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
    parser.add_argument('--train_dataset', type=str, help="The dataset of train")
    parser.add_argument('--test_dataset', type=str, help="The dataset of test")
    
    
    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.train_dataset == None:
        raise FileNotFoundError("Please Specify the training set")
    
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
    
    train_dataset = pd.read_csv(args.train_dataset)

    
    if args.test_dataset == None:
        train = train_dataset.values.tolist()
        random.shuffle(train)
        train_dataset, test_dataset = K_Fold(train, [], i=1)
    
    else:
        test_dataset = pd.read_csv(args.test_dataset)
    
    train_set = train_dataset.values.tolist()
    test_set = test_dataset.values.tolist()

    test_set = CustomDataSet(test_set)
    VT_set = CustomDataSet(train_set)

    train_size = int(len(VT_set)*0.8)
    valid_size = len(VT_set) - train_size
    train_set, valid_set = random_split(VT_set, [train_size, valid_size])

    train_dataset_load = DataLoader(train_set, batch_size=args.batch_size, collate_fn = collate_fn)        
    valid_dataset_load = DataLoader(valid_set, batch_size=args.batch_size, collate_fn = collate_fn)
    test_dataset_load = DataLoader(test_set, batch_size=args.batch_size, collate_fn = collate_fn)
    VT_dataset_load = DataLoader(VT_set, batch_size=args.batch_size, collate_fn = collate_fn)
    """ create model"""
    
    model = Rep_ConvDTI(args).cuda()
    if args.use_multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    weight_p, bias_p = [], []
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    optimizer = optim.AdamW([{"params": weight_p, "weight_decay": 1e-4}, {"params": bias_p, "weight_decay": 0}], lr=args.learning_rate)
    Loss = nn.MSELoss()
    save_path = f"./Rep_ConvDTI/"
    writer = SummaryWriter(log_dir=save_path, comment="")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    max_Acc = 0
                
    for epoch in range(1, args.train_epochs+1):
        train_pbar = tqdm(
        enumerate(
            BackgroundGenerator(train_dataset_load)),
        total=len(train_dataset_load), desc=f"{epoch}/{args.train_epochs}")
        
        train_losses_in_epoch = []
        model.train()
        for train_i, train_data in train_pbar:
            train_compounds, train_proteins, train_labels = train_data
            train_compounds = train_compounds.cuda()
            train_proteins = train_proteins.cuda()
            train_labels = train_labels.cuda()

            optimizer.zero_grad()
            
            predicted_interaction = model(train_compounds, train_proteins)
            train_loss = Loss(predicted_interaction, train_labels)
            train_losses_in_epoch.append(train_loss.item())
            train_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            optimizer.step()

        train_loss_a_epoch = np.average(train_losses_in_epoch)  # 一次epoch的平均训练loss
        writer.add_scalar('Train Loss', train_loss_a_epoch, epoch)
        
        valid_pbar = tqdm(
                enumerate(
                    BackgroundGenerator(valid_dataset_load)),
                total=len(valid_dataset_load))
        valid_losses_in_epoch = []
        model.eval()
        Y, P, S = [], [], []
        with torch.no_grad():
            for valid_i, valid_data in valid_pbar:
                compounds, proteins, labels = valid_data
                compounds = compounds.cuda()
                proteins = proteins.cuda()
                labels = torch.clamp(torch.round(labels), min=0, max=1).long().cuda()

                predicted_scores = model(compounds, proteins)
                loss = Loss(predicted_scores, labels)
                correct_labels = labels.to('cpu').data.numpy()
                predicted_labels = torch.clamp(torch.round(predicted_scores), min=0, max=1).long().to('cpu').data.numpy()
                predicted_scores = predicted_scores.to('cpu').data.numpy()

                Y.extend(correct_labels)
                P.extend(predicted_labels)
                S.extend(predicted_scores)
                valid_losses_in_epoch.append(loss.item())

        Precision = precision_score(Y, P)
        Reacll = recall_score(Y, P)
        Accuracy = accuracy_score(Y, P)
        AU_ROC = roc_auc_score(Y, S)
        tpr, fpr, _ = precision_recall_curve(Y, S)
        AU_PRC = auc(fpr, tpr)
        valid_loss_a_epoch = np.average(valid_losses_in_epoch)  

        epoch_len = len(str(args.train_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{args.train_epochs:>{epoch_len}}] ' +
                        f'train_loss: {train_loss_a_epoch:.5f} ' +
                        f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                        f'valid_AUC: {AU_ROC:.5f} ' +
                        f'valid_PRC: {AU_PRC:.5f} ' +
                        f'valid_Accuracy: {Accuracy:.5f} ' +
                        f'valid_Precision: {Precision:.5f} ' +
                        f'valid_Reacll: {Reacll:.5f} ')

        writer.add_scalar('Valid Loss', valid_loss_a_epoch, epoch)
        writer.add_scalar('Valid AUC', AU_ROC, epoch)
        writer.add_scalar('Valid AUPR', AU_PRC, epoch)
        writer.add_scalar('Valid Accuracy', Accuracy, epoch)
        writer.add_scalar('Valid Precision', Precision, epoch)
        writer.add_scalar('Valid Reacll', Reacll, epoch)    

        print(print_msg)
        if Accuracy > max_Acc:
            max_Acc = Accuracy
            torch.save(model.state_dict(), f"{save_path}best_checkpoint.pth", )
            print("save model")
            
    model.load_state_dict(torch.load(f"{save_path}best_checkpoint.pth", weights_only=True))
    model.eval()
    X_train = None
    y_train = None
    X_valid = None
    y_valid = None
    X_test = None
    y_test = None
    train_pbar = tqdm(
        enumerate(
            BackgroundGenerator(VT_dataset_load)),
        total=len(VT_dataset_load))
    for trian_i, train_data in train_pbar:
        '''data preparation '''
        trian_compounds, trian_proteins, trian_labels = train_data
        trian_compounds = trian_compounds.cuda()
        trian_proteins = trian_proteins.cuda()
        trian_labels = trian_labels.cuda()

        predicted_interaction = model.module.fussion(trian_compounds, trian_proteins)
        X = predicted_interaction.cpu().detach().numpy()
        y = trian_labels.cpu().detach().numpy()
        if X_train is None:
            X_train = X
        else:
            X_train = np.vstack((X_train, X))
        if y_train is None:
            y_train = y
        else:
            y_train = np.hstack((y_train, y))
    dtrain = xgb.DMatrix(X_train, y_train)
    xgb_model = xgb.train(plst, dtrain, num_boost_round = 200)
    xgb_model.save_model(save_path + "xgboost_model")
    """valid"""
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(test_dataset_load)),
        total=len(test_dataset_load))
    with torch.no_grad():
        for valid_i, valid_data in test_pbar:
            '''data preparation '''
            valid_compounds, valid_proteins, valid_labels = valid_data

            valid_compounds = valid_compounds.cuda()
            valid_proteins = valid_proteins.cuda()
            valid_labels = valid_labels.cuda()

            valid_scores = model.module.fussion(valid_compounds, valid_proteins)
            X = valid_scores.cpu().detach().numpy()
            y = valid_labels.cpu().detach().numpy()
            if X_valid is None:
                X_valid = X
            else:
                X_valid = np.vstack((X_valid, X))
            if y_valid is None:
                y_valid = y
            else:
                y_valid = np.hstack((y_valid, y))
        x_valid = xgb.DMatrix(X_valid)
        S = xgb_model.predict(x_valid)
        Y = np.round(y_valid)
        P = np.round(S)
        Precision = precision_score(Y, P)
        Reacll = recall_score(Y, P)
        Accuracy = accuracy_score(Y, P)
        AU_ROC = roc_auc_score(Y, S)
        tpr, fpr, _ = precision_recall_curve(Y, S)
        AU_PRC = auc(fpr, tpr)

        with open("./{}/results.txt".format(save_path), 'a+') as f:
            f.write("\n")
            f.write('Accuracy(std):{:.4f} '.format(Accuracy))
            f.write('Precision(std):{:.4f} '.format(Precision))
            f.write('Recall(std):{:.4f} '.format(Reacll))
            f.write('AUC(std):{:.4f} '.format(AU_ROC))
            f.write('PRC(std):{:.4f}\n'.format(AU_PRC))

