import torch
import torch.nn as nn
import torch.nn.functional as F



PAD_token = 0
SOS_token = 1
EOS_token = 2


def maxout(t):
    L = int(t.size(-1)/2)
    max_vals = []
    for k in range(L):
        m = torch.max(t[:, 2*k : 2*(k+1)], 1)
        max_vals.append(m[0])
    u = torch.stack(max_vals)
    return u.transpose(0, 1)


class EncoderRNN(nn.Module):
    def __init__(self, K_x, M, N):
        super(EncoderRNN, self).__init__()

        self.embedding = nn.Embedding(K_x, M)
        self.gru_forward = nn.GRU(M, N, batch_first=True)
        self.gru_backward = nn.GRU(M, N, batch_first=True)

        torch.nn.init.normal_(self.gru_forward.weight_ih_l0, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.gru_backward.weight_ih_l0, mean=0.0, std=0.01)
        torch.nn.init.orthogonal_(self.gru_forward.weight_hh_l0, gain=1)
        torch.nn.init.orthogonal_(self.gru_backward.weight_hh_l0, gain=1)
        torch.nn.init.zeros_(self.gru_forward.bias_ih_l0)
        torch.nn.init.zeros_(self.gru_forward.bias_hh_l0)
        torch.nn.init.zeros_(self.gru_backward.bias_ih_l0)
        torch.nn.init.zeros_(self.gru_backward.bias_hh_l0)

    def forward(self, x):
        """Calculates annotations and final backwards hidden state
        from input sentences"""
        Ex = self.embedding(x)
        Ex_B = torch.flip(Ex, dims=(1,))
        h_F, h_F_Tx = self.gru_forward(Ex)
        h_B, h_B_Tx = self.gru_backward(Ex_B)
        h = torch.cat((h_F, h_B), dim=2)
        return h, h_B_Tx.squeeze(0)


class AttnDecoderRNN(nn.Module):
    def __init__(self, K_y, T_y, M, N, P, L):
        super(AttnDecoderRNN, self).__init__()
        self.T_y = T_y

        self.embedding = nn.Embedding(K_y, M)
        self.Ws = nn.Linear(N, N)   # initial hidden state layer

        # alignment model (attention) layers
        self.Wa = nn.Linear(N, P)
        self.Ua = nn.Linear(2*N, P)
        self.va = nn.Linear(P, 1)

        # reset gate
        self.Wr = nn.Linear(M, N)
        self.Ur = nn.Linear(N, N)
        self.Cr = nn.Linear(2*N, N)
        # update gate
        self.Wz = nn.Linear(M, N)
        self.Uz = nn.Linear(N, N)
        self.Cz = nn.Linear(2*N, N)
        # candidate activation
        self.Wn = nn.Linear(M, N)
        self.Un = nn.Linear(N, N)
        self.Cn = nn.Linear(2*N, N)

        # target probability layers
        self.Wo = nn.Linear(L, K_y)
        self.Vo = nn.Linear(M, 2*L)
        self.Uo = nn.Linear(N, 2*L)
        self.Co = nn.Linear(2*N, 2*L)

        torch.nn.init.orthogonal_(self.Ur.weight, gain=1)
        torch.nn.init.orthogonal_(self.Uz.weight, gain=1)
        torch.nn.init.orthogonal_(self.Un.weight, gain=5/3)
        torch.nn.init.normal_(self.Wa.weight, mean=0.0, std=0.001)
        torch.nn.init.normal_(self.Ua.weight, mean=0.0, std=0.001)
        torch.nn.init.normal_(self.Ws.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.Wr.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.Cr.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.Wz.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.Cz.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.Wn.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.Cn.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.Wo.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.Vo.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.Uo.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.Co.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.va.weight)
        torch.nn.init.zeros_(self.Ws.bias)
        torch.nn.init.zeros_(self.Wa.bias)
        torch.nn.init.zeros_(self.Ua.bias)
        torch.nn.init.zeros_(self.va.bias)
        torch.nn.init.zeros_(self.Wr.bias)
        torch.nn.init.zeros_(self.Ur.bias)
        torch.nn.init.zeros_(self.Cr.bias)
        torch.nn.init.zeros_(self.Wz.bias)
        torch.nn.init.zeros_(self.Uz.bias)
        torch.nn.init.zeros_(self.Cz.bias)
        torch.nn.init.zeros_(self.Wn.bias)
        torch.nn.init.zeros_(self.Un.bias)
        torch.nn.init.zeros_(self.Cn.bias)
        torch.nn.init.zeros_(self.Wo.bias)
        torch.nn.init.zeros_(self.Vo.bias)
        torch.nn.init.zeros_(self.Uo.bias)
        torch.nn.init.zeros_(self.Co.bias)


    def forward(self, h, h_B_Tx, evaluate=False):
        """Calculates word log-probabilities (or predeicted word indices if evaluate=True)
        from annotations and final backwards hidden state of encoder"""

        B = h.size(0)
        y_im1 = torch.empty(B, 1, dtype=torch.long).fill_(SOS_token)    # create initial output word
        s_im1 = F.tanh(self.Ws(h_B_Tx))                                 # create initial hidden state

        Uah = self.Ua(h)
        y = []
        # loop over target words
        for i in range(self.T_y):

            Ey_im1 = self.embedding(y_im1).squeeze(0)   # word embedding
            c_i = self.alignment(s_im1, Uah, h)         # context
            s_i = self.GRU_step(Ey_im1, s_im1, c_i)     # hidden state

            # calculate probability distribution for y_i
            t_i = self.Vo(Ey_im1) + self.Uo(s_im1) + self.Co(c_i)
            t_i = maxout(t_i)
            Py_i = self.Wo(t_i)
            y_i = torch.argmax(Py_i, dim=1)

            y_im1 = y_i # update output
            s_im1 = s_i # update hidden state                             

            if evaluate:    # add word index batch to list y
                y.append(y_i)                                       
            else:           # add word log-probablility batch to list y
                Py_i = F.log_softmax(Py_i, dim=1)
                y.append(Py_i)                                      
            
        return torch.stack(y).transpose(0, 1)

    def alignment(self, s_im1, Uah, h):
        """Calculate context from previous hidden state, weighted annotations,
        and unweighted annotations"""

        Was_im1 = self.Wa(s_im1).unsqueeze(1)
        D = Was_im1.expand(Uah.size())
        e_i = self.va(F.tanh(D + Uah)).transpose(1,2)
        alpha_i = F.softmax(e_i, dim=-1)
        c_i = torch.bmm(alpha_i, h).squeeze(1)
        return c_i

    def GRU_step(self, Ey_im1, s_im1, c_i):
        """Calculate hidden state from previous target word embedding,
        previous hidden state, and context"""

        r_i = F.sigmoid(self.Wr(Ey_im1) + self.Ur(s_im1) + self.Cr(c_i))
        z_i = F.sigmoid(self.Wz(Ey_im1) + self.Uz(s_im1) + self.Cz(c_i))
        n_i = F.tanh(self.Wn(Ey_im1) + self.Un(r_i*s_im1) + self.Cn(c_i))
        s_i = (1-z_i)*s_im1 + z_i*n_i
        return s_i
