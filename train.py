import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys, os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

sys.path.append('MY_BERT')

from dataset.dataset import Topological_seq
# from MY_BERT.model.model import BERT
from model.prenorm_model import BERT

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default='logs', 
                        help='where to store ckpts and logs')
    
    parser.add_argument("--sup_case", type=int, default=2)
    parser.add_argument("--train_data_file_path", type=str, 
                        default='data/preprocessed_train_data.json', 
                        )
    parser.add_argument("--test_data_file_path", type=str, 
                        default='data/preprocessed_test_data.json', 
                        )
    parser.add_argument("--batch_size", type=int, default=64, )
    parser.add_argument("--nepoch", type=int, default=156,  
                        help='num of total epoches')
    parser.add_argument("--lr", type=float, default=1e-3,  
                        help='')
    
    parser.add_argument("--i_val",   type=int, default=400, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_print",   type=int, default=6000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weight", type=int, default=50000, 
                        help='frequency of weight ckpt saving')

    return parser

def train():
    parser = config_parser()
    args = parser.parse_args()

    device_id = 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    train_dataset = Topological_seq(data_file_path=args.train_data_file_path)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, drop_last=True)

    test_dataset = Topological_seq(data_file_path=args.test_data_file_path)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, drop_last=True)

    model = BERT().to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    global_step = 0
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    log_dir = os.path.join(args.basedir, 'events', TIMESTAMP + f'_sup_type={args.sup_case}')
    os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    os.makedirs(os.path.join(args.basedir, 'ckpts'), exist_ok=True)
    pbar = tqdm(total=args.nepoch * len(train_dataloader))

    for epoch in range(args.nepoch):
        for i, (seq_ids, segment_ids, start_state,  sup_states_idx, sup_states_val, all_states) in enumerate(train_dataloader):
            seq_ids, segment_ids, start_state, sup_states_idx, sup_states_val, all_states = seq_ids.to(device), segment_ids.to(device), start_state.to(device),  sup_states_idx.to(device), sup_states_val.to(device), all_states.to(device)
            # print(seq_ids.shape, segment_ids.shape, labels.shape)
            dim_of_states = start_state.shape[1]
            tar_start_state = torch.zeros(args.batch_size, 1, 768).to(device)
            tar_start_state[:, 0, :dim_of_states] = start_state
            
            output_logits = model(seq_ids, segment_ids, tar_start_state)[:, 1:]
            sup_logits = torch.stack([output_logits[i][sup_states_idx[i]] for i in range(args.batch_size)], dim=0)
            
            if args.sup_case == 0: # # 只监督最后一个状态
                loss = criterion(sup_logits[:, -1], sup_states_val[:, -1])
            elif args.sup_case == 1: # # 监督最后一帧action对应的states
                loss = criterion(sup_logits, sup_states_val)
            else: # # 监督所有状态
                loss = criterion(output_logits, all_states)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_of_trans_states = sup_states_val.shape[1]
            sup_logits = sup_logits.round()
            whole_accs = []
            for z in range(num_of_trans_states):
                whole_acc = sum([torch.equal(sup_logits[i, z], sup_states_val[i, z]) for i in range(args.batch_size)]) / args.batch_size
                whole_accs.append(whole_acc)

            writer.add_scalar('train/loss', loss.item(), global_step)
            # writer.add_scalar('train/whole_acc', whole_acc, global_step)
            for j in range(num_of_trans_states):
                writer.add_scalar(f'train/whole_acc_{j}', whole_accs[j], global_step)

            writer.add_scalar('learning rates', optimizer.param_groups[0]['lr'], global_step)
            
            pbar.update(1)
            if global_step % args.i_print  == 0:
                print(f"global_step:{global_step}, train_loss:{loss.item()}, train_acc:{whole_accs[-1]}")

            if global_step % args.i_val == 0:
                model.eval()
                with torch.no_grad():
                    test_whole_accs = []
                    total_loss = 0
                    for i, (seq_ids, segment_ids, start_state, sup_states_idx, sup_states_val, all_states) in enumerate(test_dataloader):
                        seq_ids, segment_ids, start_state, sup_states_idx, sup_states_val, all_states = seq_ids.to(device), segment_ids.to(device), start_state.to(device),  sup_states_idx.to(device), sup_states_val.to(device), all_states.to(device)
                        # print(seq_ids.shape, segment_ids.shape, labels.shape)
                        dim_of_states = start_state.shape[1]
                        tar_start_state = torch.zeros(args.batch_size, 1, 768).to(device)
                        tar_start_state[:, 0, :dim_of_states] = start_state
                        
                        output_logits = model(seq_ids, segment_ids, tar_start_state)[:, 1:]
                        sup_logits = torch.stack([output_logits[i][sup_states_idx[i]] for i in range(args.batch_size)], dim=0)
                        
                        if args.sup_case == 0: # # 只监督最后一个状态
                            loss = criterion(sup_logits[:, -1], sup_states_val[:, -1])
                        elif args.sup_case == 1: # # 监督最后一帧action对应的states
                            loss = criterion(sup_logits, sup_states_val)
                        else: # # 监督所有状态
                            loss = criterion(output_logits, all_states)


                        num_of_trans_states = sup_states_val.shape[1]
                        sup_logits = sup_logits.round()
                        whole_accs = []
                        for z in range(num_of_trans_states):
                            whole_acc = sum([torch.equal(sup_logits[i, z], sup_states_val[i, z]) for i in range(args.batch_size)]) / args.batch_size
                            whole_accs.append(whole_acc)

                        test_whole_accs.append(whole_accs)
                        total_loss += loss

                    test_loss = total_loss / len(test_dataloader)
                    test_whole_accs = np.sum(test_whole_accs, axis=0) / len(test_dataloader)

                model.train()

                writer.add_scalar('test/loss', test_loss.item(), global_step)
                # writer.add_scalar('train/whole_acc', whole_acc, global_step)
                for j in range(num_of_trans_states):
                    writer.add_scalar(f'test/whole_acc_{j}', test_whole_accs[j], global_step)

            if (global_step + 1) % args.i_weight == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'global_step': global_step,
                }, os.path.join(args.basedir, 'ckpts', f"model_{global_step}.tar"))

            global_step += 1


def reload(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    global_step = checkpoint['global_step']
    model.eval()
    # model.train()


if __name__ == '__main__':
    train()