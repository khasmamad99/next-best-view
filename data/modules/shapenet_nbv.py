from pathlib import Path
import json
import pickle

import torch
from torch.utils.data import Dataset


class ShapeNetNBV(Dataset):

    dir2idx = {
        'up'   : 0,
        'down' : 1,
        'right': 2,
        'left' : 3
    }

    idx2dir = {
        0 : 'up',
        1 : 'down',
        2 : 'right',
        3 : 'left'
    }

    def __init__(self, data_dir_path: Path, file_extension: str = 'pickle', overfit: bool = False):
        assert file_extension in ['pickle', 'json']
        self.file_extension = file_extension
        synset_ids = list(data_dir_path.glob('*'))
        self.path_to_files = list()
        for synset_id in synset_ids:
            obj_ids = list(synset_id.glob('*'))
            for obj_id in obj_ids:
                scan_sequences = list(obj_id.glob('*'))
                for scan_sequence in scan_sequences:
                    scans = list(scan_sequence.glob(f'*{file_extension}'))
                    self.path_to_files += scans
                    if overfit:
                        return

    def __len__(self):
        return len(self.path_to_files)

    def __getitem__(self, idx):
        path = self.path_to_files[idx]
        if self.file_extension == 'pickle':
            scan = self.load_pickle(path)
        elif self.file_extension == 'json':
            scan = self.load_json(path)
        partial_model = torch.Tensor(scan['partial_model']).float()
        nbv = torch.LongTensor([self.dir2idx[scan['next_view_dir']]])
        return {
            'partial_model': partial_model.unsqueeze(0),  # add channel dimension
            'nbv' : nbv
        }

    def load_json(self, path):
        with open(path, 'r') as f:
            scan = json.load(f)
        return scan

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            scan = pickle.load(f)
        return scan