'''
(c) 2023 by  Pengfei Zhang, Michael Cai, Seojin Bang, Heewook Lee, and Arizona State University.
Under a CC BY-NC-ND License
'''
import sys
import time
import os
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score, precision_score, recall_score, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path


warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2'

DATA_ROOT = Path(".")

def get_inputs(embedding_type):
    if embedding_type == 'catELMo':
        file_name = "catELMo_combined.pkl"
    elif embedding_type == 'blosum62':
        file_name = "BLOSUM62.pkl"
    elif embedding_type == 'blosum62_22_24':
        file_name = "BLOSUM62_20_22.pkl"
    elif embedding_type == 'SeqVec':
        file_name = "SeqVec.pkl"
    elif embedding_type == 'ProtBert':
        file_name = "ProtBert.pkl"
    elif embedding_type == 'catBert':
        file_name = "catBert.pkl"
    elif embedding_type == 'Doc2Vec':
        file_name = "Doc2Vec.pkl"
    elif embedding_type == 'catELMo_finetuned':
        file_name = "catELMo_finetuned.pkl"
    elif embedding_type == 'catBert_768_12_layers_mlm_nsp':
        file_name = "Small_Bert_mlm_nsp.pkl"
    elif embedding_type == 'catBert_768_12_layers_mlm':
        file_name = "Small_Bert_mlm.pkl"
    elif embedding_type == 'TCRbert':
        file_name = "TCRBert_mlm_12_layers.pkl"
    elif embedding_type == 'catBert_768_2_layers_mlm':
        file_name = "Small_Bert_mlm_2_layers.pkl"
    elif embedding_type == 'catELMo_4_layers_1024':
        file_name = "catELMo_4_layers_1024.pkl"
    elif embedding_type == 'catELMo_12_layers_1024':
        file_name = "Small_Bert_mlm_2_layers.pkl"

    # dat = dat.sample(frac=1).reset_index(drop=True)
    file_path = DATA_ROOT / file_name

    # If you run out of RAM, it is likely here
    dat = pd.read_pickle(file_path)
    # Read_pickle memory use is multiple times the dataset size

    return dat


def load_data_split(dat,split_type, seed):
    n_fold = 5
    idx_test_fold = 0
    idx_val_fold = -1
    idx_test = None
    idx_train = None
    x_pep = dat.epi.to_numpy()
    x_tcr = dat.tcr.to_numpy()

    if split_type == 'random':
        n_total = len(x_pep)
    elif split_type == 'epi':
        unique_peptides = np.unique(x_pep)
        n_total = len(unique_peptides)
    elif split_type == 'tcr':
        unique_tcrs = np.unique(x_tcr)
        n_total = len(unique_tcrs)

    np.random.seed(seed)
    idx_shuffled = np.arange(n_total)
    np.random.shuffle(idx_shuffled)

    # Determine data split from folds
    n_test = int(round(n_total / n_fold))
    n_train = n_total - n_test

    # Determine position of current test fold
    test_fold_start_index = idx_test_fold * n_test
    test_fold_end_index = (idx_test_fold + 1) * n_test

    if split_type == 'random':
        # Split data evenly among evenly spaced folds
        # Determine if there is an outer testing fold
        if idx_val_fold < 0:
            idx_test = idx_shuffled[test_fold_start_index:test_fold_end_index]
            idx_train = list(set(idx_shuffled).difference(set(idx_test)))
        else:
            validation_fold_start_index = args.idx_val_fold * n_test
            validation_fold_end_index = (args.idx_val_fold + 1) * n_test
            idx_test_remove = idx_shuffled[test_fold_start_index:test_fold_end_index]
            idx_test = idx_shuffled[validation_fold_start_index:validation_fold_end_index]
            idx_train = list(set(idx_shuffled).difference(set(idx_test)).difference(set(idx_test_remove)))
    elif split_type == 'epi':
        if idx_val_fold < 0:
            idx_test_pep = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_peptides = set(unique_peptides[idx_test_pep])
            idx_test = [index for index, pep in enumerate(x_pep) if pep in test_peptides]
            idx_train = list(set(range(len(x_pep))).difference(set(idx_test)))
        else:
            validation_fold_start_index = args.idx_val_fold * n_test
            validation_fold_end_index = (args.idx_val_fold + 1) * n_test
            idx_test_remove_pep = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_remove_peptides = set(unique_peptides[idx_test_remove_pep])
            idx_test_pep = idx_shuffled[validation_fold_start_index:validation_fold_end_index]
            test_peptides = set(unique_peptides[idx_test_pep])
            idx_test = [index for index, pep in enumerate(x_pep) if pep in test_peptides]
            idx_test_remove = [index for index, pep in enumerate(x_pep) if pep in test_remove_peptides]
            idx_train = list(set(range(len(x_pep))).difference(set(idx_test)).difference(set(idx_test_remove)))
    elif split_type == 'tcr':
        if idx_val_fold < 0:
            idx_test_tcr = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_tcrs = set(unique_tcrs[idx_test_tcr])
            idx_test = [index for index, tcr in enumerate(x_tcr) if tcr in test_tcrs]
            idx_train = list(set(range(len(x_tcr))).difference(set(idx_test)))
        else:
            validation_fold_start_index = args.idx_val_fold * n_test
            validation_fold_end_index = (args.idx_val_fold + 1) * n_test
            idx_test_remove_tcr = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_remove_tcrs = set(unique_tcrs[idx_test_remove_tcr])
            idx_test_tcr = idx_shuffled[validation_fold_start_index:validation_fold_end_index]
            test_tcrs = set(unique_tcrs[idx_test_tcr])
            idx_test = [index for index, tcr in enumerate(x_tcr) if tcr in test_tcrs]
            idx_test_remove = [index for index, tcr in enumerate(x_tcr) if tcr in test_remove_tcrs]
            idx_train = list(set(range(len(x_tcr))).difference(set(idx_test)).difference(set(idx_test_remove)))

    testData = dat.iloc[idx_test, :].sample(frac=1).reset_index(drop=True)
    trainData = dat.iloc[idx_train, :].sample(frac=1).reset_index(drop=True)


    print('================check Overlapping========================')
    print('number of overlapping tcrs: ', str(len(set(trainData.tcr).intersection(set(testData.tcr)))))
    print('number of overlapping epitopes: ', str(len(set(trainData.epi).intersection(set(testData.epi)))))

    # tcr_split testing read
    X1_test_list, X2_test_list, y_test_list = testData.tcr_embeds.to_list(), testData.epi_embeds.to_list(),testData.binding.to_list()
    X1_test, X2_test, y_test = np.array(X1_test_list), np.array(X2_test_list), np.array(y_test_list)
    # tcr_split training read
    X1_train_list, X2_train_list, y_train_list = trainData.tcr_embeds.to_list(), trainData.epi_embeds.to_list(),trainData.binding.to_list()
    X1_train, X2_train, y_train = np.array(X1_train_list), np.array(X2_train_list), np.array(y_train_list)
    return  X1_train, X2_train, y_train, X1_test, X2_test, y_test, testData, trainData


def train_(embedding_name,X1_train, X2_train, y_train, X1_test, X2_test, y_test, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Siamese network
    class TwoTower(nn.Module):
        def __init__(self, input_a, input_b):
            super().__init__()
            self.branch_a = nn.Sequential(
                nn.Linear(input_a, 2048),
                nn.BatchNorm1d(2048),
                nn.Dropout(0.3),
                nn.SiLU(),
            )
            self.branch_b = nn.Sequential(
                nn.Linear(input_b, 2048),
                nn.BatchNorm1d(2048),
                nn.Dropout(0.3),
                nn.SiLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(4096, 1024),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.3),
                nn.SiLU(),
                nn.Linear(1024, 1),
            )

        def forward(self, a, b):
            a_out = self.branch_a(a)
            b_out = self.branch_b(b)
            # combined = torch.cat([a_out, b_out, torch.abs(a_out - b_out)], dim=1)
            combined = torch.cat([a_out, b_out], dim=1)
            return self.head(combined)

    model = TwoTower(len(X1_train[0]), len(X2_train[0])).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{param_count} parameters")

    # this is like 3gb so can easily fit in GPU ram
    X1_train_tensor = torch.tensor(X1_train, dtype=torch.float32, device=device)
    del X1_train
    X2_train_tensor = torch.tensor(X2_train, dtype=torch.float32, device=device)
    del X2_train
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    del y_train
    X1_test_tensor = torch.tensor(X1_test, dtype=torch.float32, device=device)
    del X1_test
    X2_test_tensor = torch.tensor(X2_test, dtype=torch.float32, device=device)
    del X2_test
    # after moving to gpu we dont need these anymore

    n_total = len(y_train_tensor)
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val
    
    # Which indices go where
    perm = torch.randperm(n_total, device=device)
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]
    
    # Make train and val sets
    train_x1 = X1_train_tensor[train_indices]
    train_x2 = X2_train_tensor[train_indices]
    train_y = y_train_tensor[train_indices]
    val_x1 = X1_train_tensor[val_indices]
    val_x2 = X2_train_tensor[val_indices]
    val_y = y_train_tensor[val_indices]

    batch_size = 1024
    num_batches = (n_train + batch_size - 1) // batch_size

    checkpoint_filepath = embedding_name + ".pt"
    best_val = float("inf")
    epochs_since_improve = 0
    patience = 30
    max_epochs = 200

    history = {
        "train_loss": [],
        "val_loss": []
    }

    # model fit
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        indices = torch.randperm(n_train, device=device)

        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}", leave=False, mininterval=0.05) as pbar:
            for i in range(0, n_train, batch_size):
                # indices is a scrambled list of indices
                batch_idx = indices[i : i + batch_size]
                
                batch_a = train_x1[batch_idx]
                batch_b = train_x2[batch_idx]
                batch_y = train_y[batch_idx].unsqueeze(1)

                optimizer.zero_grad()
                logits = model(batch_a, batch_b)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(batch_y)
                pbar.update(1)

        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, n_val, batch_size):
                batch_a = val_x1[i : i + batch_size]
                batch_b = val_x2[i : i + batch_size]
                batch_y = val_y[i : i + batch_size].unsqueeze(1)

                logits = model(batch_a, batch_b)
                val_loss += criterion(logits, batch_y).item() * len(batch_y)

        val_loss /= n_val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"[{embedding_name}] Epoch {epoch+1:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | best_val={min(val_loss, best_val):.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            epochs_since_improve = 0
            torch.save(model.state_dict(), checkpoint_filepath)
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # torch.save(model.state_dict(), f"models/{embedding_name}.pt")
    if os.path.exists(checkpoint_filepath):
        model.load_state_dict(torch.load(checkpoint_filepath, map_location=device))

    model.eval()
    with torch.no_grad():
        test_logits = model(X1_test_tensor, X2_test_tensor)
        yhat = torch.sigmoid(test_logits).cpu().numpy().flatten()

    print('================Performance========================')
    auc_value = roc_auc_score(y_test, yhat)
    print(embedding_name+'AUC: ' + str(auc_value))


    yhat[yhat>=0.5] = 1
    yhat[yhat<0.5] = 0

    accuracy = accuracy_score(y_test, yhat)
    precision1 = precision_score(
        y_test, yhat, pos_label=1, zero_division=0)
    precision0 = precision_score(
        y_test, yhat, pos_label=0, zero_division=0)
    recall1 = recall_score(y_test, yhat, pos_label=1, zero_division=0)
    recall0 = recall_score(y_test, yhat, pos_label=0, zero_division=0)
    f1macro = f1_score(y_test, yhat, average='macro')
    f1micro = f1_score(y_test, yhat, average='micro')

    prfs = precision_recall_fscore_support(y_test,yhat, average='macro')
    precision, recall, f1, _ = prfs
    print('precision_recall_fscore_macro ' + str(prfs))
    #print('acc is '  + str(accuracy))
    #print('precision1 is '  + str(precision1))
    #print('precision0 is '  + str(precision0))
    #print('recall1 is '  + str(recall1))
    #print('recall0 is '  + str(recall0))
    #print('f1macro is '  + str(f1macro))
    #print('f1micro is '  + str(f1micro))

    metrics = {
        "name": embedding_name,
        "AUC": float(auc_value),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
    }

    return history, metrics


def main(embedding, split,fraction,seed, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu
    dat = get_inputs(embedding)
    tr_dat = dat
    tr_dat = dat.sample(frac=fraction, replace=True, random_state=seed).reset_index(drop=True) # comment this out if no fraction used
    del dat
    X1_train, X2_train, y_train, X1_test, X2_test, y_test, testData, trainData = load_data_split(tr_dat,split, seed)
    run_name = embedding + '_' + split + '_seed_' + str(seed) + '_fraction_' + str(fraction)
    history, metrics = train_(run_name, X1_train, X2_train, y_train, X1_test, X2_test, y_test, seed)

    # save history
    history_data = pd.DataFrame({
        "epoch": np.arange(len(history["train_loss"])), 
        "train_loss": history["train_loss"], 
        "val_loss": history["val_loss"], 
    })
    history_data["embedding"] = embedding
    history_data["split"] = split
    history_data["seed"] = seed
    history_data["fraction"] = fraction

    history_data.to_csv(f"loss_{run_name}.csv", index=False)

    # add split to metrics
    metrics["split"] = split

    return metrics



if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--embedding', type=str,help='elmo or blosum62')
    #parser.add_argument('--split', type=str,help='random, tcr or epi')
    #parser.add_argument('--gpu', type=str)
    #parser.add_argument('--fraction', type=float, default=1.0)
    #parser.add_argument('--seed', type=int, default=42)
    #args = parser.parse_args()
    #main(args.embedding, args.split, args.fraction, args.seed, args.gpu)
    embedding = "catELMo"
    splits = ["tcr", "epi"]
    fraction = 1.0
    num_seeds = 5
    gpu = "0"

    seeds = list(range(42, 42 + num_seeds))
    all_metrics = []

    for split in splits:
        for seed in seeds:
            print(f"Embedding {embedding} split {split} seed {seed}")
            metrics = main(embedding, split, fraction, seed, gpu)
            all_metrics.append(metrics)

    lines = []
    for m in all_metrics:
        lines.append(f"{m['name']}")
        for metric in ["AUC", "Precision", "Recall", "F1"]:
            lines.append(f"  {metric}: {m[metric]:.4f}")
        lines.append("")
    
    lines.append("================================")
    lines.append("")

    for split in splits:
        metrics_in_split = [m for m in all_metrics if m["split"] == split]
        lines.append(f"{split} stats")
        for metric in ["AUC", "Precision", "Recall", "F1"]:
            vals = np.array([m[metric] for m in metrics_in_split], dtype=float)
            mean = vals.mean()
            std = vals.std(ddof=1)
            lines.append(f"  {metric} mean: {mean:.4f}")
            lines.append(f"  {metric} std: {std:.4f}")
        lines.append("")

    output = "\n".join(lines)

    print(output)

    with open(f"results_{embedding}.txt", "w") as file_out:
        file_out.write(output)