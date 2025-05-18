import torch
import torch.nn as nn
import torch.nn.functional as F

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # Parametri del modello: pesi e bias
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def sample_h(self, v):
        """Campiona layer nascosto h dato lo stato visibile v"""
        wx = F.linear(v, self.W, self.h_bias)  # h = v @ W.T + b
        p_h_given_v = torch.sigmoid(wx)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, h):
        """Campiona layer visibile v dato lo stato nascosto h"""
        wx = F.linear(h, self.W.t(), self.v_bias)  # v = h @ W + b
        p_v_given_h = torch.sigmoid(wx)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def free_energy(self, v):
        """Calcola l'energia libera (negativa della log-probabilità non normalizzata)"""
        vbias_term = torch.matmul(v, self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = torch.sum(F.softplus(wx_b), dim=1)
        return -vbias_term - hidden_term

    def forward(self, v, k=1):
        """Contrastive Divergence (CD-k): aggiornamento dei pesi"""
        v0 = v
        for _ in range(k):
            p_h, h = self.sample_h(v)
            p_v, v = self.sample_v(h)
            v = v.detach()  # stop gradient per CD

        return v0, v

    def contrastive_divergence(self, v0, vk, lr=0.01):
        """Aggiorna i pesi usando CD-1"""
        ph0, h0 = self.sample_h(v0)
        phk, hk = self.sample_h(vk)

        # Aggiornamento pesi
        self.W.data += lr * (torch.matmul(h0.t(), v0) - torch.matmul(hk.t(), vk)) / v0.size(0)
        self.v_bias.data += lr * torch.mean(v0 - vk, dim=0)
        self.h_bias.data += lr * torch.mean(ph0 - phk, dim=0)