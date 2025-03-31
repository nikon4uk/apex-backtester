import pandas as pd
import vectorbt as vbt
from typing import Dict, List


from strategies.base import StrategyBase
from core.metrics import base_metrics


class MultiTimeframeMomentumStrategy(StrategyBase):
    def __init__(
            self, 
            price_data: Dict[str, pd.DataFrame],
            fast_period: int = 14,
            slow_period: int = 50,
            timeframes: List[str] = ['5min', '15min'],
            **kwargs
        ):
        super().__init__(price_data, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.timeframes = timeframes
    
    def resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        return data.resample(timeframe).last()
    
    def generate_signals(self) -> Dict[str, pd.DataFrame]:
        signals = {}
        
        for symbol, data in self.price_data.items():
            resampled_signals = []
            
            for tf in self.timeframes:
                resampled_data = self.resample_data(data, tf)
                fast_ma = resampled_data['close'].rolling(self.fast_period).mean()
                slow_ma = resampled_data['close'].rolling(self.slow_period).mean()
                
                signal = (fast_ma > slow_ma).astype(int).reindex(data.index).ffill()
                resampled_signals.append(signal)
                
            final_signal = pd.concat(resampled_signals, axis=1).mean(axis=1).round().astype(int)
            
            entries = final_signal.shift(1) < final_signal
            exits = final_signal.shift(1) > final_signal
            
            signals[symbol] = pd.DataFrame({'entries': entries, 'exits': exits}, index=data.index)
        
        return signals
    
    def run_backtest(self) -> Dict[str, vbt.Portfolio]:
        signals = self.generate_signals()
        
        close_df = pd.DataFrame({symbol: data['close'] for symbol, data in self.price_data.items()})
        entries_df = pd.DataFrame({symbol: signals[symbol]['entries'] for symbol in signals})
        exits_df = pd.DataFrame({symbol: signals[symbol]['exits'] for symbol in signals})
        
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

