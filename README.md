# Physics-Inspired Portfolio Optimization with Simulated Annealing

> **A Monte Carlo approach to portfolio optimization using financial data and spin systems.**

---

## 📝 Description

This project explores a simple approach to **portfolio optimization** inspired by **Ising spin models** from statistical physics.

Each asset in the portfolio is represented by a binary variable (`+1` = Buy, `-1` = Sell), and we optimize a cost function (energy) that balances **expected return** and **risk (covariance)**. The optimization is performed via **Simulated Annealing**, a stochastic algorithm inspired by thermodynamic cooling processes.

---

## 🧠 Motivation

Classical portfolio optimization (e.g., mean-variance theory) assumes convexity and continuous weights. However, in real-world scenarios:
- You often **cannot short** or allocate fractions of assets,
- The search space becomes **combinatorially hard**.

This project models the problem as a **binary optimization** task and solves it using tools from **statistical physics**.

---

## 🧮 Methodology

### 1. Data Acquisition
- Historical price data for ~50 real assets (from Yahoo Finance)
- Preprocessing to compute **returns** and **covariance matrix**

### 2. Ising-like Model

#### Mathematical Formulation

The portfolio optimization problem is mapped to an Ising model where:

- **Spin Variables**: Each asset is represented by a binary spin variable sᵢ ∈ {-1, +1}
  - sᵢ = +1: Buy the asset
  - sᵢ = -1: Sell the asset

- **Expected Returns (μ)**: 
  - μᵢ represents the expected return of asset i
  - Computed as the mean of historical returns
  - Higher μᵢ indicates higher expected future returns

- **Covariance Matrix (Σ)**:
  - Σᵢⱼ represents the covariance between assets i and j
  - Measures how assets move together
  - Diagonal elements (Σᵢᵢ) represent individual asset variances
  - Off-diagonal elements represent co-movements between assets

#### Energy Function Mapping

The Hamiltonian (energy function) is constructed as:

```math
E(s) = -½ sᵀ J s - hᵀ s
```

Where:
- **J** (Interaction Matrix):
  - J = -λΣ, where λ is a risk aversion parameter
  - Negative sign because we want to minimize risk
  - Jᵢⱼ represents the interaction strength between assets i and j
  - Strong positive correlations lead to stronger interactions

- **h** (External Field):
  - h = μ, the vector of expected returns
  - Represents the "bias" towards buying high-return assets
  - Higher hᵢ means stronger preference to buy asset i

#### Physical Interpretation

1. **Risk Term** (-½ sᵀ J s):
   - Quadratic term representing portfolio risk
   - Minimized when assets with positive correlation have opposite spins
   - Encourages diversification

2. **Return Term** (-hᵀ s):
   - Linear term representing expected returns
   - Maximized when high-return assets have sᵢ = +1
   - Encourages buying profitable assets

#### Optimization Process

The Simulated Annealing algorithm:
1. Starts with a random portfolio configuration
2. Iteratively proposes changes to single assets (spin flips)
3. Accepts changes based on Metropolis criterion:
   ```math
   P(accept) = min(1, exp(-ΔE/T))
   ```
   where:
   - ΔE is the energy change
   - T is the "temperature" parameter that decreases over time
4. Gradually reduces temperature to find the ground state (optimal portfolio)


