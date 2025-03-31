import pandas as pd
import vectorbt as vbt
from typing import Dict


from strategies.base import StrategyBase
from core.metrics import base_metrics


class SMACrossoverStrategy(StrategyBase):
    def __init__(
            self, 
            price_data: Dict[str, pd.DataFrame], 
            short_window: int = 50, 
            long_window: int = 200,
            **kwargs,
        ):
        super().__init__(price_data, **kwargs)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self) -> Dict[str, pd.DataFrame]:
        """
        Generating signals for the SMA Crossover strategy for all symbols.
        
        :return: Dictionary of DataFrames with columns 'entries' and 'exits' for each symbol
        """
        signals = {}
        
        for symbol, data in self.price_data.items():
            short_sma = data['close'].rolling(self.short_window).mean()
            long_sma = data['close'].rolling(self.long_window).mean()

            entries = (short_sma > long_sma).astype(int)
            exits = (short_sma < long_sma).astype(int)

            signals[symbol] = pd.DataFrame({
                'entries': entries,
                'exits': exits
            }, index=data.index)
            
        return signals

    def run_backtest(self) -> Dict[str, vbt.Portfolio]:
        """
        Running a backtest for the SMA Crossover strategy for all symbols simultaneously.
        
        :return: Dictionary of portfolios with backtest results for each symbol
        """
        signals = self.generate_signals()

        # Створюємо словник з даними для vectorbt
        close_df = pd.DataFrame({symbol: data['close'] for symbol, data in self.price_data.items()})
        entries_df = pd.DataFrame({symbol: signals[symbol]['entries'] for symbol in signals})
        exits_df = pd.DataFrame({symbol: signals[symbol]['exits'] for symbol in signals})
        
        # Запускаємо бектест для всіх пар одночасно через vectorbt
        self.portfolio = vbt.Portfolio.from_signals(
            close=close_df,
            entries=entries_df,
            exits=exits_df,
            freq=pd.infer_freq(next(iter(self.price_data.values())).index),
            fees=self.fees,
            slippage=self.slippage,
            init_cash=self.init_cash
        )
        
        return self.portfolio

    def get_metrics(self) -> dict:
        """
        Calculating metrics for the SMA Crossover strategy for each symbol.
        
        :return: Dictionary with metrics for each symbol
        """
        if self.portfolio is None:
            self.run_backtest()
    
        overall_metrics = self.portfolio.stats(metrics=base_metrics, group_by=True)
        grouped_metrics = self.portfolio.stats(metrics=base_metrics, agg_func=None)
        
     
        return {
            'overall': overall_metrics.to_dict(),
            'by_symbol': grouped_metrics.to_dict()
        }
