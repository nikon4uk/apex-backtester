import os
from typing import List, Type, Dict


import pandas as pd
import vectorbt as vbt


from strategies.base import StrategyBase
from core.visualizer import Visualizer


class Backtester:
    def __init__(self, strategies: List[StrategyBase]):
        self.strategies = strategies
        self.visualizer = Visualizer()
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

    def run_backtest(self) -> List[dict]:
        """Main method to run backtest for all strategies"""
        all_results = []
        
        for strategy in self.strategies:
            strategy_results = self._run_single_strategy(strategy)
            all_results.extend(strategy_results)
            
        self._save_all_results(all_results)
        metrics_df = pd.DataFrame(all_results)
        self.visualizer.generate_comparison_plots(metrics_df)
        return all_results

    def _run_single_strategy(self, strategy: StrategyBase) -> List[dict]:
        """Run backtest for a single strategy"""
        strategy_name = strategy.__class__.__name__
        print(f"\nRunning backtest for {strategy_name} strategy...")
        
        portfolio = strategy.run_backtest()
        strategy_metrics = strategy.get_metrics()
        
        self._save_strategy_results(strategy_name, strategy_metrics)
        self._generate_strategy_visuals(strategy_name, portfolio)
        return self._format_results(strategy_name, strategy_metrics)

    def _generate_strategy_visuals(self, strategy_name: str, portfolio: vbt.Portfolio):
        """Generate all visualizations for a single strategy"""
        # self.visualizer.generate_equity_curve(portfolio, strategy_name)
        self.visualizer.generate_bubble_heatmap(portfolio, strategy_name)
        self.visualizer.generate_heatmap(portfolio, strategy_name)
        self.visualizer.generate_html_report(portfolio, strategy_name)

    def _format_results(self, strategy_name: str, results: dict) -> List[dict]:
        """Format results into a unified structure"""
        formatted = []
        
        # Process symbol-specific metrics
        for metric, values in results['by_symbol'].items():
            for symbol, value in values.items():
                formatted.append({
                    'strategy': strategy_name,
                    'symbol': symbol,
                    'metric': metric,
                    'value': value
                })
        
        # Process overall metrics
        for metric, value in results['overall'].items():
            formatted.append({
                'strategy': strategy_name,
                'symbol': 'overall',
                'metric': metric,
                'value': value
            })
            
        return formatted

    def _save_strategy_results(self, strategy_name: str, results: dict):
        """Save metrics for single strategy in wide format"""
        metrics = list(results['overall'].keys())
        data = {'metric': metrics}
        
        # Add overall metrics
        data['overall'] = [results['overall'][m] for m in metrics]
        
        # Add per-symbol metrics
        for symbol in next(iter(results['by_symbol'].values())).keys():
            data[symbol] = [results['by_symbol'][m].get(symbol, None) for m in metrics]
        
        pd.DataFrame(data).to_csv(
            os.path.join(self.results_dir, f"{strategy_name}_metrics.csv"),
            index=False
        )

    def _save_all_results(self, results: List[dict]):
        """Save all results in long format"""
        pd.DataFrame(results).to_csv(
            os.path.join(self.results_dir, "all_strategies_metrics.csv"),
            index=False
        )