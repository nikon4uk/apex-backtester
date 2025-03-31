from typing import Dict


import pandas as pd
import vectorbt as vbt

from strategies.base import StrategyBase
from core.metrics import base_metrics

class VWAPReversionStrategy(StrategyBase):
    """
        VWAP Reversion Intraday Strategy
        
        :param price_data: Словник з DataFrame для кожного символу
        :param lookback_window: Розмір вікна для розрахунку VWAP (в барах)
        :param deviation_threshold: Відхилення від VWAP для входу (у стандартних відхиленнях)
        :param tp_multiplier: Множник take-profit відносно deviation_threshold
        :param sl_multiplier: Множник stop-loss відносно deviation_threshold
    """
    def __init__(
            self, price_data: Dict[str, pd.DataFrame], 
            lookback_window: int = 50,
            deviation_threshold: float = 1.5,
            tp_multiplier: float = 1.0,
            sl_multiplier: float = 1.5,
            **kwargs
        ):
        
        super().__init__(price_data, **kwargs)
        self.lookback_window = lookback_window
        self.deviation_threshold = deviation_threshold
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier

    def calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Розраховує VWAP для даних"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        volume = data['volume']
        return typical_price.rolling(self.lookback_window).apply(
            lambda x: (x * volume[x.index]).sum() / volume[x.index].sum(),
            raw=False
        )

    def generate_signals(self) -> Dict[str, pd.DataFrame]:
        """Генерує сигнали входу/виходу на основі відхилення від VWAP"""
        signals = {}
        
        for symbol, data in self.price_data.items():
            # Розраховуємо VWAP та стандартне відхилення
            vwap = self.calculate_vwap(data)
            std_dev = data['close'].rolling(self.lookback_window).std()
            
            # Визначаємо рівні відхилення
            upper_band = vwap + self.deviation_threshold * std_dev
            lower_band = vwap - self.deviation_threshold * std_dev
            
            # Сигнали входу
            entries_short = (data['close'] > upper_band)  # Продаж при відхиленні вверх
            entries_long = (data['close'] < lower_band)   # Купівля при відхиленні вниз
            
            # Рівні TP/SL
            take_profit = vwap.copy()
            
            # Create stop_loss based on position direction
            stop_loss = vwap.copy()
            mask_long = entries_long
            mask_short = entries_short
            
            # For long positions
            stop_loss[mask_long] = vwap[mask_long] + (self.sl_multiplier * std_dev[mask_long] * -1)
            
            # For short positions
            stop_loss[mask_short] = vwap[mask_short] + (self.sl_multiplier * std_dev[mask_short] * 1)
            
            signals[symbol] = pd.DataFrame({
                'entries_long': entries_long.astype(int),
                'entries_short': entries_short.astype(int),
                'vwap': vwap,
                'take_profit': take_profit,
                'stop_loss': stop_loss
            }, index=data.index)
            
        return signals

    def run_backtest(self) -> vbt.Portfolio:
        """Запускає бектест стратегії для всіх символів одночасно"""
        signals = self.generate_signals()
        
        # Підготовка даних для vectorbt
        close_df = pd.DataFrame({s: d['close'] for s, d in self.price_data.items()})
        
        # Створюємо DataFrame для всіх сигналів
        entries_long_df = pd.DataFrame({s: sig['entries_long'] for s, sig in signals.items()})
        entries_short_df = pd.DataFrame({s: sig['entries_short'] for s, sig in signals.items()})
        
        # Розраховуємо рівні TP/SL у відсотках для всіх символів
        tp_pct = pd.DataFrame({
            s: (sig['take_profit'] - close_df[s]) / close_df[s] 
            for s, sig in signals.items()
        })
        
        sl_pct = pd.DataFrame({
            s: (sig['stop_loss'] - close_df[s]) / close_df[s]
            for s, sig in signals.items()
        })
        
        # Для коротких позицій інвертуємо значення TP/SL
        tp_pct_short = -tp_pct.where(entries_short_df.astype(bool))
        sl_pct_short = -sl_pct.where(entries_short_df.astype(bool))
        
        # Комбінуємо TP/SL для довгих і коротких позицій
        final_tp = tp_pct.where(entries_long_df.astype(bool), tp_pct_short)
        final_sl = sl_pct.where(entries_long_df.astype(bool), sl_pct_short)
        
        # Запускаємо бектест для всіх символів одночасно
        self.portfolio = vbt.Portfolio.from_signals(
            close=close_df,
            entries=entries_long_df,
            short_entries=entries_short_df,
            tp_stop=final_tp,
            sl_stop=final_sl,
            freq=pd.infer_freq(next(iter(self.price_data.values())).index),
            accumulate=False,
            fees=self.fees,
            slippage=self.slippage,
            init_cash=self.init_cash
        )
        
        return self.portfolio

    def get_metrics(self) -> dict:
        """Повертає метрики стратегії"""
        if self.portfolio is None:
            self.run_backtest()
            
        overall_metrics = self.portfolio.stats(metrics=base_metrics, group_by=True)
        grouped_metrics = self.portfolio.stats(metrics=base_metrics, agg_func=None)
        
        return {
            'overall': overall_metrics.to_dict(),
            'by_symbol': grouped_metrics.to_dict()
        }
