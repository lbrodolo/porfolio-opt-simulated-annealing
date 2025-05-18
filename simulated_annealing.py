from utils import energy
import numpy as np

def simulated_annealing(J, h, steps=100_000, T_init=0.1, T_final=0.01, verbose=True):
    N = len(h)
    s = np.random.choice([-1, 1], size=N)
    best_s = s.copy()
    best_E = energy(s, J, h)
    energies = []
    accepted = 0

    for step in range(steps):
        T = T_init * (T_final / T_init) ** (step / steps)  # raffreddamento esponenziale

        i = np.random.randint(N)
        s_new = s.copy()
        s_new[i] *= -1

        dE = energy(s_new, J, h) - energy(s, J, h)
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            s = s_new
            accepted += 1
            E = energy(s, J, h)
            if E < best_E:
                best_E = E
                best_s = s.copy()

        if verbose and step % (steps // 100) == 0:
            print(f"Step {step}: Energy = {energy(s, J, h):.6f}")

        energies.append(energy(s, J, h))

    acceptance_rate = accepted / steps
    return np.array(energies), best_s, acceptance_rate

