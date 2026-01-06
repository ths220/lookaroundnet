from torch.utils.data import IterableDataset
import numpy as np
from tqdm import tqdm
import random

class EDFDataset(IterableDataset):

    def __init__(self, data, block_size, look_behind_size, look_ahead_size, overlap_size, sampling_rate, overlap = False, eval = False):
        super(EDFDataset).__init__()
        
        self.block_size = block_size * sampling_rate
        self.look_behind_size = look_behind_size * sampling_rate
        self.look_ahead_size = look_ahead_size * sampling_rate
        self.overlap_size = overlap_size * sampling_rate if overlap else 0
        self.eval = eval
        self.data = data

    def __iter__(self):
        block_ids = self._get_block_ids()

        for id in tqdm(block_ids):
            rec_id, index = id
            eeg_block, labels_block = self._get_block(rec_id, index)
            yield eeg_block, labels_block, rec_id, index
        
    def _get_balanced_block_ids(self):
        # Balance blocks by seizure activity
        n_sz_blocks, n_non_sz_blocks, n_mixed_blocks = (20000, 20000, 20000)

        sz_ids = []
        non_sz_ids = []
        mixed_ids = []

        patient_indices = {}

        # Sort blocks by patient
        for file_path, rec in self.data.items():
            p = rec["patient"]
            if p not in patient_indices:
                patient_indices[p] = {"sz": 0, "non_sz": 0, "mixed": 0}
            for category, indices in rec["indices"].items():
                patient_indices[p][category] += len(indices)

        n_sz_patients = sum(1 for p, counts in patient_indices.items() if counts["sz"] > 0)
        n_non_sz_patients = sum(1 for p, counts in patient_indices.items() if counts["non_sz"] > 0)
        n_mixed_patients = sum(1 for p, counts in patient_indices.items() if counts["mixed"] > 0)
        
        # Select blocks balanced by patient, segment and seizure activity
        for file_path, rec in self.data.items():
            if rec['indices']['sz']:
                n_sz_samples_per_range = int(n_sz_blocks / n_sz_patients / patient_indices[rec['patient']]['sz'])
                sz_ids.extend((file_path, idx) for range in rec['indices']['sz'] for idx in random.choices(range, k=max(1, n_sz_samples_per_range)))
            if rec['indices']['non_sz']:
                n_non_sz_samples_per_range = int(n_non_sz_blocks / n_non_sz_patients / patient_indices[rec['patient']]['non_sz'])
                non_sz_ids.extend((file_path, idx) for range in rec['indices']['non_sz'] for idx in random.choices(range, k=max(1, n_non_sz_samples_per_range)))
            if rec['indices']['mixed']:
                n_mixed_samples_per_range = int(n_mixed_blocks / n_mixed_patients / patient_indices[rec['patient']]['mixed'])
                mixed_ids.extend((file_path, idx) for range in rec['indices']['mixed'] for idx in random.choices(range, k=max(1, n_mixed_samples_per_range)))
        
        ids = sz_ids + non_sz_ids + mixed_ids
        random.shuffle(ids)

        return ids

    def _get_all_block_ids(self):
        # Get all overlapping recording blocks
        step_size = self.block_size - self.overlap_size
        ids = []

        for file_path, rec in self.data.items():
            for category, indices in rec["indices"].items():
                ids.extend((file_path, idx) for segment in indices for idx in segment if idx % step_size == 0)
        
        ids.sort(key=lambda x: (x[0], x[1]))

        return ids
        
    def _get_block_ids(self):
        if self.eval:
            return self._get_all_block_ids()
        else:
            return self._get_balanced_block_ids()

    def _get_block(self, rec_id, idx):
        # Get EEG segment by index
        rec = self.data[rec_id]

        start_index = idx - self.look_behind_size
        end_index = idx + self.block_size + self.look_ahead_size
        eeg_block = rec['eeg'][:,max(0, start_index):min(rec['eeg'].shape[1], end_index)]
        labels_block = rec['targets'][idx:idx + self.block_size]

        # Pad
        if (start_index < 0):
            eeg_block = self._pad(eeg_block, eeg_block.shape[1] - start_index, 1, False)
        if (end_index > rec['targets'].shape[0]):
            eeg_block = self._pad(eeg_block, eeg_block.shape[1] + (end_index - rec['targets'].shape[0]), 1, True)

        if labels_block.shape[0] < self.block_size:
            labels_block = self._pad(labels_block, self.block_size, 0, True)

        return eeg_block, labels_block

    def _pad(self, matrix, target_width, dim, end):
        # Zero-pad
        pad_width = [(0, 0)] * matrix.ndim
        if end:
            pad_width[dim] = (0, max(0, target_width - matrix.shape[dim]))
        else:
            pad_width[dim] = (max(0, target_width - matrix.shape[dim]), 0)

        padded_matrix = np.pad(matrix, pad_width, mode='constant', constant_values=0)
        return padded_matrix