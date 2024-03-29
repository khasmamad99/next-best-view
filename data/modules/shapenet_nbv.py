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

    def __init__(
        self, 
        data_dir_path: Path, 
        file_extension: str = 'pickle', 
        select_objects_start: int = 0,
        select_objects_end: int = None,
    ):
        assert file_extension in ['pickle', 'json']
        if select_objects_end:
            assert select_objects_start < select_objects_end
        self.file_extension = file_extension
        synset_ids = sorted(list(data_dir_path.glob('*')))
        self.path_to_files = list()
        for synset_id in synset_ids:
            obj_ids = sorted(list(synset_id.glob('*')))[select_objects_start:select_objects_end]
            for obj_id in obj_ids:
                scan_sequences = list(obj_id.glob('*'))
                for scan_sequence in scan_sequences:
                    scans = list(scan_sequence.glob(f'*{file_extension}'))
                    self.path_to_files += scans
    
    
    def __len__(self):
        return len(self.path_to_files)

    def __getitem__(self, idx):
        path = self.path_to_files[idx]
        if self.file_extension == 'pickle':
            scan = self.load_pickle(path)
        elif self.file_extension == 'json':
            scan = self.load_json(path)
        # wrap in torch.Tensor and add channel dimension
        partial_model = torch.Tensor(scan['partial_model']).float().unsqueeze(0)
        nbv = torch.LongTensor([self.dir2idx[scan['next_view_dir']]])
        return partial_model, nbv

    def load_json(self, path):
        with open(path, 'r') as f:
            scan = json.load(f)
        return scan

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            scan = pickle.load(f)
        return scan