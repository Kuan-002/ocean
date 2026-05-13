# Credit: koriavinash1. VisualDebates:modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ModulatorNet(nn.Module):
    def __init__(self, input_size: int, output_size: int, use_gpu: bool):
        super(ModulatorNet, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.final = nn.Linear(output_size, output_size)
        if use_gpu:
            self.cuda()

    def forward(self, x):
        x = F.relu(self.fc(x))
        return F.relu(self.final(x))

class StateRNN(nn.Module):
    def __init__(self, input_size: int, modulator_output_size: int, rnn_hidden: int, use_gpu: bool, rnn_type: str='RNN'):
        super(StateRNN, self).__init__()
        self.input_size = input_size
        self.modulator_output_size = modulator_output_size
        self.rnn_hidden = rnn_hidden
        self.use_gpu = use_gpu
        self.rnn_type = rnn_type
        
        self.modulator = ModulatorNet(input_size, modulator_output_size, use_gpu)
        
        if rnn_type == 'RNN':
            self.rnn = nn.RNNCell(modulator_output_size, rnn_hidden, bias=True, nonlinearity='relu')
        elif rnn_type == 'GRU':
            self.rnn = nn.GRUCell(modulator_output_size, rnn_hidden, bias=True)
        
        if use_gpu:
            self.cuda()
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, s_t: torch.Tensor, h_t_prev: torch.Tensor):
        g_t = self.modulator(s_t)
        h_t = self.rnn(g_t, h_t_prev)
        return h_t
    
    def init_hidden(self, batch_size: int):
        dtype = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        h = torch.zeros(batch_size, self.rnn_hidden)
        h_t = Variable(h).type(dtype)
        return h_t
    
    def eval(self):
        super().eval()
        self.modulator.eval()
    
    def train(self, mode=True):
        super().train(mode)
        self.modulator.train(mode)

class Classifier(nn.Module):
    """ A simple MLP classifier attached to a player. """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, use_gpu: bool):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        if use_gpu:
            self.cuda()

    def forward(self, h_t):
        h_t = F.relu(self.fc1(h_t))
        h_t = self.fc2(h_t)
        return h_t

class RNNClassifier(nn.Module):
    """ A RNN and Classifier pair for tracking a hidden state and making predictions. """
    def __init__(self, modulator_input_size: int, rnn_input_size: int, hidden_state_size: int, hidden_layer_size: int, output_size: int, use_gpu: bool):
        super(RNNClassifier, self).__init__()
        self.rnn = StateRNN(modulator_input_size, rnn_input_size, hidden_state_size, use_gpu, rnn_type='GRU')
        self.classifier = Classifier(hidden_state_size, hidden_layer_size, output_size, use_gpu)
        self.hidden_state = None
        
        if use_gpu:
            self.cuda()

    def forward_rnn(self, s_t: torch.Tensor):
        self.hidden_state = self.rnn(s_t, self.hidden_state)
        return self.hidden_state
    
    def forward_classifier(self):
        return self.classifier(self.hidden_state)

    def reset_hidden(self, batch_size: int):
        self.hidden_state = self.rnn.init_hidden(batch_size)
        return self.hidden_state

    def eval(self):
        super().eval()
        self.rnn.eval()
        self.classifier.eval()

    def train(self, mode=True):
        super().train(mode)
        self.rnn.train(mode)
        self.classifier.train(mode)

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int, use_gpu: bool):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, action_dim)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        if use_gpu:
            self.cuda()

    def forward(self, x: torch.Tensor):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        logits = self.fc3(x)
        return self.softmax(logits)
    
class RNNPolicyNetwork(nn.Module):
    """ A RNN and PolicyNetwork pair for tracking a hidden state and making predictions. """
    def __init__(self, modulator_input_size: int, rnn_input_size: int, hidden_state_size: int, hidden_layer_size: int, output_size: int, num_indices: int, use_gpu: bool):
        super(RNNPolicyNetwork, self).__init__()
        self.pos_embed = nn.Embedding(num_indices, modulator_input_size)
        self.rnn = StateRNN(modulator_input_size, rnn_input_size, hidden_state_size, use_gpu, rnn_type='RNN')
        self.policy = PolicyNetwork(hidden_state_size, hidden_layer_size, output_size, use_gpu)
        self.hidden_state = None

        if use_gpu:
            self.cuda()

    def forward_rnn(self, s_t, positions):
        emb = self.pos_embed(positions)
        s_t = s_t + emb
        self.hidden_state = self.rnn(s_t, self.hidden_state)
        return self.hidden_state
    
    def forward_policy(self):
        return self.policy(self.hidden_state)

    def reset_hidden(self, batch_size):
        self.hidden_state = self.rnn.init_hidden(batch_size)
        return self.hidden_state

    def eval(self):
        super().eval()
        self.rnn.eval()
        self.policy.eval()

    def train(self, mode=True):
        super().train(mode)
        self.rnn.train(mode)
        self.policy.train(mode)

class Baseline(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, use_gpu: bool):
        super(Baseline, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        
        if use_gpu:
            self.cuda()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.activation(x)
        return self.fc2(x).squeeze(-1)
