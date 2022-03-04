from cmath import log
from numpy import dtype
from torch.utils.data import DataLoader, Dataset
import json
import torch

class Topological_seq(Dataset):
    def __init__(self, data_file_path) -> None:
        super().__init__()
        with open(data_file_path, 'r') as load_f:
            self.seqs=json.load(load_f)
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq = torch.tensor(self.seqs[idx]['seq'], dtype=torch.long)
        segment = torch.tensor(self.seqs[idx]['segment'], dtype=torch.long)
        # label = torch.tensor(self.seqs[idx]['label'])
        start_state = torch.tensor(self.seqs[idx]['start_state'], dtype=torch.float32)
        sup_states_idx = torch.tensor(self.seqs[idx]['sup_states_idx'], dtype=torch.long)
        sup_states_val = torch.tensor(self.seqs[idx]['sup_states_val'], dtype=torch.float32)
        all_states = torch.tensor(self.seqs[idx]['all_states'], dtype=torch.float32)
        return seq, segment, start_state, sup_states_idx, sup_states_val, all_states


if __name__ == '__main__':
    dataset = Topological_seq(data_file_path="data/preprocessed_test_data.json")
    dataloader = DataLoader(dataset, batch_size = 8, shuffle=True, drop_last=True)
    print(len(dataloader))
    for i, (seq, segment, start_state, sup_states_idx, sup_states_val, all_states) in enumerate(dataloader):
        print(seq.shape, segment.shape, start_state.shape)
        # # sup_states_idx: B x num_of_supstates 
        # # sup_states_val: B x num_of_supstates x stateLen(6)
        print(sup_states_idx.shape, sup_states_val.shape, all_states.shape)

        output_logits = torch.rand(sup_states_val.shape[0], seq.shape[1], 6)
        sup_logits = torch.stack([output_logits[i][sup_states_idx[i]] for i in range(8)], dim=0)
        print('sup_logits:', sup_logits.shape)