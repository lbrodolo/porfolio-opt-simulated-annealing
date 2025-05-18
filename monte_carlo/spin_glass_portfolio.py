import numpy as np

class SpinGlassPortfolio:
    def __init__(self, num_assets, J_matrix=None, h=None, seed=42):
        self.num_assets = num_assets
        np.random.seed(seed)

        self.s = np.random.choice([-1, 1], size=num_assets)  # Stato iniziale: -1/1 = vendi/compri

        if J_matrix is None:
            self.J = np.random.randn(num_assets, num_assets) * 0.1  # Interazioni casuali
            self.J = (self.J + self.J.T) / 2  # Simmetrica
            np.fill_diagonal(self.J, 0)
        else:
            self.J = J_matrix

        if h is None:
            self.h = np.random.randn(num_assets) * 0.05  # Bias su ogni asset
        else:
            self.h = h

    def energy(self, config=None):
        if config is None:
            config = self.s
        E = -0.5 * np.dot(config, np.dot(self.J, config)) - np.dot(self.h, config)
        return E

    def metropolis_step(self, T=1.0):
        i = np.random.randint(0, self.num_assets)
        s_new = self.s.copy()
        s_new[i] *= -1  # Flip
        delta_E = self.energy(s_new) - self.energy(self.s)
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            self.s = s_new

    def run(self, steps=1000, T=1.0):
        energies = []
        for _ in range(steps):
            self.metropolis_step(T)
            energies.append(self.energy())
        return np.array(energies), self.s