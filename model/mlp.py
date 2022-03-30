from argparse import Action
import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal

dim_of_states = 7
d_hidden = 128

device_id = 0
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin

class MLP_Unit(nn.Module):
    '''input: state: B x dim_of_states, action: B x 1, output: B x dim_of_states'''
    def __init__(self, d_hidden, dim_of_states):
        super(MLP_Unit, self).__init__()
        self.d_hidden = d_hidden
        self.dim_of_states = dim_of_states
        self.linear1 = linear(dim_of_states + 1, d_hidden)
        self.linear2 = linear(d_hidden, d_hidden)
        self.linear3 = linear(d_hidden, d_hidden)
        self.linear4 = linear(d_hidden, d_hidden)
        self.linear5 = linear(d_hidden, d_hidden)
        self.linear6 = linear(d_hidden, dim_of_states)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))
        x = torch.relu(self.linear5(x))
        x = self.linear6(x)
        return x

class MLP_Model(nn.Module):
    '''input: action_sequence: B x T x 1, initial state: B x 7
    output: B x T x 7'''
    def __init__(self, d_hidden=128, dim_of_states=7):
        super(MLP_Model, self).__init__()
        self.mlp_unit = MLP_Unit(d_hidden, dim_of_states)
        self.dim_of_states = dim_of_states
    
    def forward(self, action_sequence, initial_state):
        B, T, _ = action_sequence.shape
        output_lst = []
        state = initial_state
        for i in range(T):
            action = action_sequence[:, i, :]
            state = self.mlp_unit(state, action) + state
            output_lst.append(state)
        output = torch.stack(output_lst, dim=1)
        return output
        

if __name__ == '__main__':
    model = MLP_Model(d_hidden, dim_of_states)
    model.to(device)
    print(model)

    action_seq = torch.randn(2, 10, 1).to(device)
    initial_state = torch.randn(2, dim_of_states).to(device)

    output = model(action_seq, initial_state)
    print(output.shape)