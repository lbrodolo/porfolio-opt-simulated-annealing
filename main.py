
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_market_data
from simulated_annealing import simulated_annealing



def main():
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'UNH', 'V',
        'MA', 'HD', 'LLY', 'MRK', 'PEP', 'ABBV', 'AVGO', 'COST', 'DIS', 'KO',
        'WMT', 'XOM', 'CVX', 'BAC', 'PFE', 'INTC', 'T', 'QCOM', 'TXN', 'CRM',
        'MDT', 'IBM', 'GE', 'PM', 'AMGN', 'LOW', 'NKE', 'MCD', 'LIN', 'SBUX',
        'ORCL', 'ACN', 'HON', 'CMCSA', 'DHR', 'ABT', 'BMY', 'NEE', 'ADBE', 'GS'
    ]

    J, h, labels = get_market_data(tickers)
    print("Assets:", labels)

    energies, best_config, acc_rate = simulated_annealing(J, h, steps=500_000, T_init=1.5, T_final=1e-5)
    print(f"\nAcceptance rate: {acc_rate:.4f}")

    portfolio = pd.DataFrame({
        'Asset': labels,
        'Decision': ['Buy' if s == 1 else 'Sell' for s in best_config],
        'Expected Return (h)': h
    }).sort_values(by='Expected Return (h)', ascending=False)

    print("\nFinal Portfolio Configuration:")
    print(portfolio)

    plt.figure(figsize=(10, 5))
    plt.plot(energies, lw=0.5)
    plt.xlabel('Step')
    plt.ylabel('Risk (Energy)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()