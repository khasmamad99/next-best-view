from pathlib import Path
import json

import torch
from torch.utils.data import Dataset

dir2idx = {
    'up'   : 0,
    'down' : 1,
    'right': 2,
    'left' : 3
}

class ShapeNetNBV(Dataset):

    def __init__(self, data_dir_path: Path):
        synset_ids = list(data_dir_path.glob('*'))
        self.path_to_files = list()
        for synset_id in synset_ids:
            obj_ids = list(synset_id.glob('*'))
            for obj_id in obj_ids:
                scan_sequences = list(obj_id.glob('*'))
                for scan_sequence in scan_sequences:
                    scans = list(scan_sequence.glob('*'))
                    self.path_to_files += scans

    def __len__(self):
        return len(self.path_to_files)

    def __getitem__(self, idx):
        path = self.path_to_files[idx]
        with open(path, 'r') as f:
            scan = json.load(f)
            partial_model = torch.Tensor(scan['partial_model']).float()
            nbv = torch.LongTensor([dir2idx[scan['next_view_dir']]])
            return {
                'partial_model': partial_model,
                'nbv' : nbv
            }

