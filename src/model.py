import torch
import torch.nn as nn
import torch.nn.functional as F


class SE_detect(nn.Module):
    def __init__(self, which, input_size, device, args):
        super().__init__()

        if which in ('rnn', 'lstm', 'gru'):
            self.tl_processing_layer = RNNcustom(input_size, args.hid_dim, args.layers, args.hid_dim, which)
        elif which == 'tf':
            self.tl_processing_layer = Transformer(input_size, args.hid_dim, 35, args.layers, 8, args.hid_dim, 0, device)
        elif which == 'retain':
            self.tl_processing_layer = RETAIN(input_size, input_size, args.layers, args.hid_dim)
            
        self.pi_processing_layer = nn.Sequential(
            nn.Linear(21, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.se_detection_layer = nn.Sequential(
            nn.Linear(64 + args.hid_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.onlytl = nn.Sequential(
            nn.Linear(args.hid_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, tl, demo):
        tl = self.tl_processing_layer(tl)
        demo = self.pi_processing_layer(demo)
        tldemo = torch.cat([tl, demo], dim=1)
        out = self.se_detection_layer(tldemo)

        return out


class SE_detect_forshap(nn.Module):
    def __init__(self, which, input_size, device, args, deep=True):
        super().__init__()\
        
        self.deep = deep

        if which in ('rnn', 'lstm', 'gru'):
            self.tl_processing_layer = RNNcustom(input_size, args.hid_dim, args.layers, args.hid_dim, which)
        elif which == 'tf':
            self.tl_processing_layer = Transformer(input_size, args.hid_dim, 35, args.layers, 8, args.hid_dim, 0, device)
        elif which == 'retain':
            self.tl_processing_layer = RETAIN(input_size, input_size, args.layers, args.hid_dim)
            
        self.pi_processing_layer = nn.Sequential(
            nn.Linear(21, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.se_detection_layer = nn.Sequential(
            nn.Linear(64 + args.hid_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.onlytl = nn.Sequential(
            nn.Linear(args.hid_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, input):
        N = input.shape[0]
        tl, demo = torch.Tensor(input[:, :-21]).view(N, 35, 23), torch.Tensor(input[:, -21:])
        tl = self.tl_processing_layer(tl)
        demo = self.pi_processing_layer(demo)
        tldemo = torch.cat([tl, demo], dim=1)
        out = self.se_detection_layer(tldemo)

        return out if self.deep else out.detach().numpy()

    

class Transformer(nn.Module):
    def __init__(self, 
                 input_dim,
                 hid_dim, 
                 length,
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()

        self.device = device
        
        self.tl_embedding = nn.Linear(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.fc1 = nn.Linear(length * hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim // 2)
        self.fc3 = nn.Linear(hid_dim // 2, 2)
        self.fclayer = nn.Sequential(self.fc1, self.fc2, self.fc3)
        
        
    def forward(self, tl):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        
        batch_size = tl.shape[0]
        src_len = tl.shape[1]
        
        tl = self.tl_embedding(tl)
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        
        tl = self.dropout((tl * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            tl, attention = layer(tl)
            
        return tl[:, -1, :]
    

class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src, attention = self.self_attention(src, src, src)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src, attention


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention
        
        
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x


class RNNcustom(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 out_dim,
                 rnn='lstm'
                 ):
        super().__init__()

        if rnn == 'lstm':
            self.RNN = nn.LSTM(input_size=input_dim, 
                                hidden_size=hid_dim, 
                                num_layers=n_layers)
        elif rnn == 'gru':
            self.RNN = nn.GRU(input_size=input_dim, 
                                hidden_size=hid_dim, 
                                num_layers=n_layers)
        elif rnn == 'rnn':
            self.RNN = nn.RNN(input_size=input_dim, 
                                hidden_size=hid_dim, 
                                num_layers=n_layers)
        
    def forward(self, tl):
        x = self.RNN(tl)[0]

        return x[:, -1, :]



class RETAIN(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 out_dim
                 ):
        super().__init__()

        self.EMB = nn.Linear(input_dim, hid_dim)
        self.LSTM_a = nn.LSTM(input_size=hid_dim, 
                              hidden_size=hid_dim, 
                              num_layers=n_layers)
        self.LSTM_b = nn.LSTM(input_size=hid_dim,
                              hidden_size=hid_dim,
                              num_layers=n_layers)
        self.W_alpha = nn.Linear(hid_dim, 1)
        self.W_beta = nn.Linear(hid_dim, hid_dim)
        self.b_alpha = nn.Parameter(torch.randn(1))
        self.b_beta = nn.Parameter(torch.randn(hid_dim))
        self.W = nn.Linear(hid_dim, out_dim)
        self.b = nn.Parameter(torch.randn(out_dim))

        self.fc2 = nn.Linear(hid_dim, hid_dim // 2)
        self.fc3 = nn.Linear(hid_dim // 2, 2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fclayer = nn.Sequential(self.fc2,
                                     self.relu,
                                     self.fc3)
        
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, tl):
        # Size [batch size * hid_dim]
        v = self.EMB(tl)

        # Size [batch size * length]
        g = self.LSTM_a(v)[0]
        e = self.W_alpha(g) + self.b_alpha
        alpha = self.softmax(e.view(e.shape[0], -1))

        # Size [batch size * length * hid_dim]
        h = self.LSTM_b(v)[0]
        beta = self.tanh(self.W_beta(h) + self.b_beta)
        
        # Size [batch size * hid_dim]
        c = alpha.view(alpha.shape[0], -1, 1) * beta * v
        c = torch.sum(c, dim=1)

        # Size [batch size * out_dim]
        y = self.W(c) + self.b

        return y
