from abc import ABC, abstractmethod
from typing import Dict


import pandas as pd
import vectorbt as vbt


class StrategyBase(ABC):
    def __init__(
        self,
        price_data: Dict[str, pd.DataFrame],
        fees: float = 0.001,          # Глобальні значення за замовчуванням
        slippage: float = 0.0005,
        init_cash: float = 10000
    ):
        self.price_data = price_data
        self.fees = fees
        self.slippage = slippage
        self.init_cash = init_cash
    
    @abstractmethod
    def generate_signals(self) -> Dict[str, pd.DataFrame]:
        """Generate buy/sell signals for all symbols"""
        pass

    @abstractmethod
    def run_backtest(self) -> Dict[str, vbt.Portfolio]:
        """Run backtest for all symbols simultaneously"""
        pass

    @abstractmethod
    def get_metrics(self, portfolio: vbt.Portfolio) -> Dict:
        """Calculate metrics for a specific symbol"""
        pass