# Quantum-Inspired Portfolio Optimization with Simulated Annealing

> **A Monte Carlo approach to portfolio optimization using financial data, spin systems, and realistic investment constraints.**

---

## 📝 Description

This project explores a novel approach to **portfolio optimization** inspired by **Ising spin models** from statistical physics.

Each asset in the portfolio is represented by a binary variable (`+1` = Buy, `-1` = Sell), and we optimize a cost function (energy) that balances **expected return** and **risk (covariance)**. The optimization is performed via **Simulated Annealing**, a stochastic algorithm inspired by thermodynamic cooling processes.

In the advanced version, we impose **real-world constraints** to make the optimization problem more realistic and challenging — an ideal setting to test the utility of neural network-based heuristics in the future.

---

## 🧠 Motivation

Classical portfolio optimization (e.g., mean-variance theory) assumes convexity and continuous weights. However, in real-world scenarios:
- You often **cannot short** or allocate fractions of assets,
- There are **budget and diversification constraints**,
- The search space becomes **combinatorially hard**.

This project models the problem as a **binary optimization** task and solves it using tools from **statistical physics**.

---

## 🧮 Methodology

### 1. Data Acquisition
- Historical price data for ~50 real assets (from Yahoo Finance)
- Preprocessing to compute **returns** and **covariance matrix**

### 2. Ising-like Model
Energy function:
```math
E(s) = -½ sᵀ J s - hᵀ s
