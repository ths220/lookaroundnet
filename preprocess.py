import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import mne

def get_montage(eeg_raw, channel_names, active, reference):
    # Montage
    active = [a.lower() for a in active]
    reference = [r.lower() for r in reference]
    name_to_idx = {ch: i for i, ch in enumerate(channel_names)}
    eeg_montage = np.zeros((len(active), eeg_raw.shape[1]))

    # Adjacent electrodes to approximate missing Fz or Pz
    electrode_adjacency = {'fz': ['f3', 'f4'], 'pz': ['p3', 'p4']}

    
    def get_signal(ch):
        if ch in name_to_idx:
            return eeg_raw[name_to_idx[ch]]
        return np.mean([eeg_raw[name_to_idx[adj]] for adj in electrode_adjacency[ch]], axis=0)
    
    # Compute montage
    for i, (act, ref) in enumerate(zip(active, reference)):
        if f"{act}-{ref}" in name_to_idx:
            eeg_montage[i] = eeg_raw[name_to_idx[f"{act}-{ref}"]]
        else:
            eeg_montage[i] = get_signal(act) - get_signal(ref)

    return eeg_montage

def filter_and_resample(eeg, sampling_rate, target_sfreq, high_pass, low_pass, notch):
    # High-pass and low-pass filtering
    eeg_filtered = mne.filter.filter_data(eeg, sampling_rate, high_pass, low_pass, verbose=False)

    # Notch filtering
    if notch:
        eeg_filtered = mne.filter.notch_filter(eeg_filtered, sampling_rate, notch, verbose=False)
    
    # Downsample
    eeg_processed = mne.filter.resample(eeg_filtered, 1, sampling_rate / target_sfreq, verbose=False)

    return eeg_processed

def get_labels(file_path, n_samples, sampling_rate, block_size):
    # Read annotation files
    seizures = pd.read_csv(os.path.splitext(file_path)[0] + ".csv_bi", comment='#')
    seizures = seizures[seizures['label'] != 'bckg']

    targets = np.zeros(n_samples)
    indices = {'sz': [], 'non_sz': [], 'mixed': []}
    current_idx = 0
    rec_end_idx = n_samples - block_size * sampling_rate + 1

    # Sort recording blocks by seizure activity
    for _, seizure in seizures.iterrows():
        non_sz_start_idx = current_idx

        sz_start_idx = int(seizure['start_time'] * sampling_rate)
        sz_end_idx = min(int(seizure['stop_time'] * sampling_rate), rec_end_idx)

        mixed_1_start_idx = max(sz_start_idx - block_size * sampling_rate + 1, current_idx)
        mixed_2_start_idx = max(sz_end_idx - block_size * sampling_rate + 1, current_idx)

        targets[sz_start_idx:sz_end_idx] = 1

        if mixed_1_start_idx > non_sz_start_idx:
            indices['non_sz'].append(range(non_sz_start_idx, mixed_1_start_idx))
        if sz_start_idx > mixed_1_start_idx:
            indices['mixed'].append(range(mixed_1_start_idx, sz_start_idx))
        if mixed_2_start_idx > sz_start_idx:
            indices['sz'].append(range(sz_start_idx, mixed_2_start_idx))
        if sz_end_idx > mixed_2_start_idx:
            if (indices['mixed'] and (mixed_2_start_idx < indices['mixed'][-1][0])):
                indices['mixed'][-1] = range(indices['mixed'][-1][0], sz_end_idx)
            else:
                indices['mixed'].append(range(mixed_2_start_idx, sz_end_idx))
        
        current_idx = sz_end_idx


    if rec_end_idx > current_idx:
        indices['non_sz'].append(range(current_idx, rec_end_idx))

    
    return targets, indices

def load_edf(file_path):
    # Read data from edf
    raw_edf = mne.io.read_raw_edf(file_path, infer_types = True, verbose=False)
    sampling_rate = int(raw_edf.info['sfreq'])
    eeg_raw = raw_edf.get_data(units = {'eeg': 'uV'})
    channel_names = [channel.split("-")[0].lower() for channel in raw_edf.ch_names]

    return eeg_raw, sampling_rate, channel_names
    

def get_eeg(eeg_raw, high_pass, low_pass, notch, target_sampling_rate, sampling_rate, channel_names, active, reference):
    # Pre-process EEG
    eeg_montage_raw = get_montage(eeg_raw, channel_names, active, reference)
    eeg_processed = filter_and_resample(eeg_montage_raw, sampling_rate, target_sampling_rate, high_pass, low_pass, notch)

    return eeg_processed

def preprocess_edf(file_path, high_pass, low_pass, notch, target_sampling_rate, active, reference, block_size):
    # Read edf
    eeg_raw, sampling_rate, channel_names = load_edf(file_path)

    # Pre-process
    eeg = get_eeg(eeg_raw, high_pass, low_pass, notch, target_sampling_rate, sampling_rate, channel_names, active, reference)

    # Assign labels
    targets, indices = get_labels(file_path, eeg.shape[1], target_sampling_rate, block_size)
    
    return file_path, {
        "eeg": eeg,
        "targets": targets,
        "indices": indices,
        "patient": os.path.basename(file_path).split("_")[0]
    }


def preprocess_edfs(file_paths, high_pass, low_pass, notch, sampling_rate, active, reference, block_size):
    data = {}

    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(preprocess_edf, file_path, high_pass, low_pass, notch, sampling_rate, active, reference, block_size)
            for file_path in file_paths
        ]
        for future in tqdm(futures):
            file_path, result = future.result()
            data[file_path] = result

    return data