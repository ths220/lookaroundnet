import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import schedulefree
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import f1_score
from timescoring import annotations, scoring
from preprocess import preprocess_edfs
from dataset import EDFDataset
from models import LookAroundNet


def get_data_loader(file_paths, overlap, eval, config): 
    # Preprocess EDF files
    data = preprocess_edfs(file_paths, config['high_pass'], config['low_pass'], config['notch'], config['sampling_rate'], config['active'], config['reference'], config['block_size'])
    
    # Build dataloader
    dataset = EDFDataset(data, config['block_size'], config['look_behind_size'], config['look_ahead_size'], config['overlap_size'], config['sampling_rate'], overlap, eval)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'])
    
    return data_loader

def get_model(config, device):
    # Initialize model
    max_seq_len = int((config['block_size']+config['look_behind_size']+config['look_ahead_size'])*config['sampling_rate']/config['patch_size'])
    model = LookAroundNet(len(config['active']), config['patch_size'], config['patch_size'], config['embedding_dim'], config['num_heads'], config['num_transformer_layers'], config['num_cnn_layers'], config['hidden_dim'], 2, max_seq_len)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model.to(device)

    return model

def compute_scores(tp, fp, ref_true, duration):
    # Sensitivity
    sensitivity = tp / ref_true if ref_true > 0 else np.nan

    # Precision
    total_pred = tp + fp
    precision = tp / total_pred if total_pred > 0 else np.nan

    # F1 Score
    if np.isnan(sensitivity) or np.isnan(precision):
        f1 = np.nan
    elif (sensitivity + precision) == 0: 
        f1 = 0
    else:
        f1 = 2 * sensitivity * precision / (sensitivity + precision)

    # False positive rate (per day)
    fp_rate = fp / (duration / 3600 / 24)

    return sensitivity, precision, f1, fp_rate

def print_scores(title, tp, fp, ref_true, duration):
    sensitivity, precision, f1, fp_rate = compute_scores(
        tp=tp,
        fp=fp,
        ref_true=ref_true,
        duration=duration,
    )

    print(
        f"# {title}\n"
        f"- Sensitivity : {sensitivity:.2f}\n"
        f"- Precision   : {precision:.2f}\n"
        f"- F1-score    : {f1:.2f}\n"
        f"- FP/24h      : {fp_rate:.2f}\n"
    )


def report_szcore(preds, targets):
    scores = {
        "exam_id": [],
        "duration": [],
        "tp_sample": [],
        "fp_sample": [],
        "ref_true_sample": [],
        "tp_event": [],
        "fp_event": [],
        "ref_true_event": [],
    }

    #Loop through recordings
    for rec_id, rec_targets_raw in targets.items():
        rec_targets = annotations.Annotation(rec_targets_raw, 1)
        rec_preds = annotations.Annotation(preds[rec_id], 1)

        sample_score = scoring.SampleScoring(rec_targets, rec_preds)
        event_score = scoring.EventScoring(rec_targets, rec_preds)

        scores["exam_id"].append(rec_id)
        scores["duration"].append(len(rec_targets_raw))

        scores["tp_sample"].append(sample_score.tp)
        scores["fp_sample"].append(sample_score.fp)
        scores["ref_true_sample"].append(sample_score.refTrue)

        scores["tp_event"].append(event_score.tp)
        scores["fp_event"].append(event_score.fp)
        scores["ref_true_event"].append(event_score.refTrue)

    scores_df = pd.DataFrame(scores)

    total_duration = scores_df["duration"].sum()

    # Sample-based scores
    print_scores(
        title="Sample scoring",
        tp=scores_df["tp_sample"].sum(),
        fp=scores_df["fp_sample"].sum(),
        ref_true=scores_df["ref_true_sample"].sum(),
        duration=total_duration,
    )

    # Event-based scores
    print_scores(
        title="Event scoring",
        tp=scores_df["tp_event"].sum(),
        fp=scores_df["fp_event"].sum(),
        ref_true=scores_df["ref_true_event"].sum(),
        duration=total_duration,
    )

    return scores_df


def run(config):
    device = torch.device("cuda")

    # Load Data
    test_data_loader = get_data_loader(config['test_file_paths'], True, True, config)
    val_data_loader = get_data_loader(config['val_file_paths'], False, True, config)
    train_data_loader = get_data_loader(config['train_file_paths'], False, False, config)
    
    # Initalize model
    model = get_model(config, device)
    
    # Training setup
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_f1_val = 0
    best_model_state = None

    # Train
    for epoch in range(config['num_epochs']):
        model.train()
        optimizer.train()
        optimizer.zero_grad()

        train_loss = 0
        train_all_sz_probs = []
        train_all_targets = []

        for inputs, targets, rec_id, rec_index in train_data_loader:
            targets = targets.view(targets.size(0), -1).mean(dim=-1).round()
            inputs, targets = inputs.to(device, torch.float), targets.to(device, torch.long)

            output = model(inputs)
            
            loss = criterion(output, targets)

            train_loss += loss.item()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            train_all_targets += targets.flatten().tolist()
            train_all_sz_probs += output.softmax(dim=1)[:,1].flatten().tolist()
        
        train_all_preds = [1 if prob >= config['threshold'] else 0 for prob in train_all_sz_probs]
        f1_train = f1_score(train_all_targets, train_all_preds)
        
        # Validate
        model.eval()
        optimizer.eval()

        val_loss = 0
        val_all_sz_probs = []
        val_all_targets = []

        with torch.no_grad():
            for inputs, targets, rec_id, rec_index in val_data_loader:
                targets = targets.view(targets.size(0), -1).mean(dim=-1).round()
                inputs, targets = inputs.to(device, torch.float), targets.to(device, torch.long)
                
                output = model(inputs)

                loss = criterion(output, targets)
                val_loss += loss.item()

                val_all_targets += targets.flatten().tolist()
                val_all_sz_probs += output.softmax(dim=1)[:,1].flatten().tolist()
                
        val_all_preds = [1 if prob >= config['threshold'] else 0 for prob in val_all_sz_probs]
        f1_val = f1_score(val_all_targets, val_all_preds)

        print(f"Epoch: {epoch}, Val F1: {f1_val}, Train F1: {f1_train}")

        # Store the best model weights
        if f1_val > best_f1_val:
            best_f1_val = f1_val
            best_model_state = model.module.state_dict()
    
    # Test
    model.module.load_state_dict(best_model_state)
    model.eval()

    test_all_sz_probs = defaultdict(lambda: defaultdict(list))
    test_all_targets = defaultdict(dict)

    for inputs, targets, rec_ids, rec_indices in test_data_loader:
        targets = targets.view(targets.size(0), config['block_size'], -1).mean(dim=-1).round()
        inputs, targets = inputs.to(device, torch.float), targets.to(device, torch.long)

        outputs = model(inputs).softmax(dim = 1)[:, 1].tolist()

        for output, target, rec_id, rec_idx in zip(outputs, targets, rec_ids, rec_indices):
            target = target.detach().cpu()

            for j in range(config['block_size']):
                test_all_sz_probs[rec_id][rec_idx + j].append(output)
                test_all_targets[rec_id][rec_idx + j] = target[j].item()
    
    # Model predictions
    final_sz_preds = {
        rec_id: [int(np.mean(test_all_sz_probs[rec_id][idx]) >= config['threshold'])
                for idx in sorted(test_all_sz_probs[rec_id])]
        for rec_id in test_all_sz_probs
    }
    
    # True labels
    final_targets = {
        rec_id: [test_all_targets[rec_id][idx]
                for idx in sorted(test_all_targets[rec_id])]
        for rec_id in test_all_targets
    }

    # SzCORE results
    report_szcore(final_sz_preds, final_targets)

def main():

    config = {
        # Data paths
        "train_file_paths": glob.glob("./data/train/**/*.edf", recursive=True),
        "val_file_paths": glob.glob("./data/val/**/*.edf", recursive=True),
        "test_file_paths": glob.glob("./data/test/**/*.edf", recursive=True),

        # Filtering parameters
        "high_pass": 0.5,
        "low_pass": 64,
        "notch": 60,
        "sampling_rate": 128,

        # EEG montage
        "active": [
            "Fp2", "F4", "C4", "P4", "Fp1", "F3",
            "C3", "P3", "Fp2", "F8", "T4", "T6",
            "Fp1", "F7", "T3", "T5", "Fz", "Cz"
        ],
        "reference": [
            "F4", "C4", "P4", "O2", "F3", "C3",
            "P3", "O1", "F8", "T4", "T6", "O2",
            "F7", "T3", "T5", "O1", "Cz", "Pz"
        ],

        # Model & windowing parameters
        "block_size": 16,
        "look_behind_size": 32,
        "look_ahead_size": 32,
        "overlap_size": 14,
        "patch_size": 48,
        "embedding_dim": 96,
        "num_heads": 3,
        "num_transformer_layers": 3,
        "num_cnn_layers": 2,
        "hidden_dim": 384,

        # Training parameters
        "learning_rate": 5e-4,
        "batch_size": 512,
        "num_epochs": 200,
        "threshold": 0.85,
    }

    run(config)

if __name__ == '__main__':
    main()