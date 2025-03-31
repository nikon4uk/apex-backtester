import os
import pandas as pd
import vectorbt as vbt
from typing import Dict, Any, List

class Visualizer:
    def __init__(self, results_dir: str = "results"):

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.4f}'.format)

        self.results_dir = results_dir
        os.makedirs(f"{results_dir}/plots", exist_ok=True)
        os.makedirs(f"{results_dir}/reports", exist_ok=True)

    def generate_equity_curve(self, portfolio: vbt.Portfolio, strategy_name: str):
        """Генерує криві доходності для кожного символу окремо"""
        try:
            # Для кожного символу окремо
            for symbol in portfolio.close.columns:
                try:
                    symbol_pf = portfolio[symbol]
                    fig = symbol_pf.plot()
                    fig.write_image(
                        f"{self.results_dir}/plots/{strategy_name}_{symbol}_equity.png",
                        engine="kaleido"
                    )
                except Exception as e:
                    print(f"Error generating equity curve for {symbol}: {e}")
            
            # Сумарний графік (якщо більше одного символу)
            if len(portfolio.close.columns) > 1:
                try:
                    fig = portfolio.value().vbt.plot(title="Combined Portfolio Value")
                    fig.write_image(
                        f"{self.results_dir}/plots/{strategy_name}_combined_equity.png",
                        engine="kaleido"
                    )
                except Exception as e:
                    print(f"Error generating combined equity curve: {e}")
        except Exception as e:
            print(f"Error in equity curve generation: {e}")

    def generate_heatmap(self, portfolio: vbt.Portfolio, strategy_name: str):
        """Генерує теплокарту доходностей по місяцях"""
        try:
            monthly_returns = []
            symbols = portfolio.close.columns.tolist()
            
            for symbol in symbols:
                # Отримуємо доходи для конкретного символу
                symbol_returns = portfolio[symbol].returns()
                monthly_rets = symbol_returns.resample('ME').sum()
                monthly_returns.append(monthly_rets)
            
            # Створюємо DataFrame для теплокарти
            heatmap_df = pd.concat(monthly_returns, axis=1)
            heatmap_df.columns = symbols
            
            # Теплокарта через Plotly Express
            import plotly.express as px
            fig = px.imshow(
                heatmap_df.T,
                labels=dict(x="Month", y="Symbol", color="Return"),
                aspect="auto",
                title="Monthly Returns Heatmap"
            )
            fig.update_xaxes(tickangle=45)
            fig.write_image(
                f"{self.results_dir}/plots/{strategy_name}_heatmap.png",
                engine="kaleido"
            )
        except Exception as e:
            print(f"Error generating heatmap: {e}")

    def generate_html_report(self, portfolio: vbt.Portfolio, strategy_name: str):
        """Генерує HTML-звіт з основними метриками"""
        try:
            # Отримуємо статистику для кожного символу окремо
            stats_list = []
            for symbol in portfolio.close.columns:
                stats = portfolio[symbol].stats()
                stats['Symbol'] = symbol
                stats_list.append(stats)
            
            # Комбінуємо всі статистики
            all_stats = pd.concat(stats_list, axis=1).T
            
            # Зберігаємо як HTML
            with open(f"{self.results_dir}/reports/{strategy_name}.html", "w", encoding='utf-8') as f:
                f.write("<html><body>")
                f.write(f"<h1>Strategy Report: {strategy_name}</h1>")
                f.write("<h2>Performance Metrics by Symbol</h2>")
                f.write(all_stats.to_html())
                
                # Додаємо загальну статистику (якщо більше одного символу)
                if len(portfolio.close.columns) > 1:
                    f.write("<h2>Combined Portfolio Metrics</h2>")
                    combined_stats = portfolio.stats()
                    f.write(combined_stats.to_frame().to_html())
                
                f.write("</body></html>")
        except Exception as e:
            print(f"Error generating HTML report: {e}")

    def generate_bubble_heatmap(self, portfolio: vbt.Portfolio, strategy_name: str):
        """Генерує інтерактивну бульбашкову теплокарту без помилок"""
        try:
            import plotly.express as px
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            # 1. Збираємо дані для кожної пари
            metrics = []
            for symbol in portfolio.close.columns:
                pf = portfolio[symbol]
                trades = pf.trades
                
                metrics.append({
                    'Symbol': symbol,
                    'Total Return (%)': pf.total_return() * 100,
                    'Trade Count': len(trades),
                    'Avg Trade Duration': trades.duration.mean(),
                    'Sharpe': pf.sharpe_ratio(),
                    'Win Rate (%)': trades.win_rate() * 100
                })
            
            df = pd.DataFrame(metrics).sort_values('Total Return (%)', ascending=False)

            # 2. Створюємо фіктивні координати для сітки
            df['GridX'] = 1  # Фіктивна координата X для всіх точок
            
            # 3. Створюємо інтерактивну теплокарту
            fig = px.scatter(
                df,
                x='GridX',  # Використовуємо спеціально створену колонку
                y="Symbol",
                size="Trade Count",
                color="Total Return (%)",
                color_continuous_scale="RdYlGn",
                hover_name="Symbol",
                hover_data={
                    'Total Return (%)': ':.1f',
                    'Trade Count': True,
                    'Sharpe': ':.2f',
                    'Win Rate (%)': ':.1f',
                    'Avg Trade Duration': ':.1f days',
                    'GridX': False  # Приховуємо фіктивну вісь
                },
                size_max=40,
                title=f"{strategy_name} Performance Matrix<br>"
                    f"<sup>Color: Total Return | Size: Trade Count</sup>"
            )

            # 4. Кастомізація вигляду
            fig.update_layout(
                xaxis_visible=False,
                xaxis_title=None,
                yaxis_title="Trading Pairs",
                coloraxis_colorbar=dict(title="Return %"),
                uniformtext_minsize=8,
                uniformtext_mode='hide',
                height=max(400, 100 + len(df) * 15),  # Мінімальна висота 400px
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial"
                )
            )

            # 5. Збереження
            os.makedirs(self.results_dir, exist_ok=True)
            fig.write_html(
                f"{self.results_dir}/bubble_heatmap_{strategy_name}.html",
                full_html=False
            )
            
            print(f"Successfully generated bubble heatmap for {strategy_name}")
            return True

        except Exception as e:
            print(f"Error generating bubble heatmap: {str(e)}")
            return False
            
    def generate_comparison_plots(self, metrics_df: pd.DataFrame):
        """Генерує порівняльні графіки для всіх груп метрик"""
        try:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            
            # Фільтруємо тільки загальні метрики
            overall_metrics = metrics_df[metrics_df['symbol'] == 'overall']
            
            # Визначаємо групи метрик
            metric_groups = {
                'Returns': ['Total Return [%]', 'Win Rate [%]', 'Expectancy'],
                'Risk': ['Max Drawdown [%]', 'Avg Loss [%]'],
                'Ratios': ['Sharpe Ratio', 'Sortino Ratio', 'Profit Factor']
            }
            
            # Створюємо графік з підграфіками
            fig = make_subplots(
                rows=1, 
                cols=len(metric_groups),
                subplot_titles=list(metric_groups.keys()),
                horizontal_spacing=0.15
            )
            
            # Додаємо дані для кожної групи
            for col, (group_name, metrics) in enumerate(metric_groups.items(), 1):
                available_metrics = [
                    m for m in metrics 
                    if m in overall_metrics['metric'].unique()
                ]
                
                for metric in available_metrics:
                    metric_data = overall_metrics[overall_metrics['metric'] == metric]
                    
                    fig.add_trace(
                        go.Bar(
                            x=metric_data['strategy'],
                            y=metric_data['value'],
                            name=metric,
                            text=metric_data['value'].round(2),
                            textposition='inside',
                            insidetextanchor='middle'
                        ),
                        row=1,
                        col=col
                    )
            
            # Налаштування вигляду
            fig.update_layout(
                height=500,
                width=350 * len(metric_groups),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.05),
                margin=dict(l=40, r=40, b=80, t=60, pad=4),
                font=dict(size=12),
            )
            
            # Додаткові налаштування підграфіків
            for i in range(1, len(metric_groups) + 1):
                fig.update_xaxes(tickangle=30, row=1, col=i)
                fig.update_yaxes(title_text="Value", row=1, col=i)
            
            # Зберігаємо графік
            os.makedirs(self.results_dir, exist_ok=True)
            fig.write_image(
                f"{self.results_dir}/plots/strategy_comparison.png",
                engine="kaleido",
                scale=2
            )
            
            print("Successfully generated comparison plots")
            return True
            
        except Exception as e:
            print(f"Error generating comparison plots: {str(e)}")
            return False