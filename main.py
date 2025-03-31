import os
import asyncio

from dotenv import load_dotenv

from core.dataloader.binance import BinanceDataLoader
from core.backtester import Backtester
from strategies.sma_cross import SMACrossoverStrategy
from strategies.vwarp_reversion import VWAPReversionStrategy
from strategies.multi_timeframe_momentum import MultiTimeframeMomentumStrategy

from config import GLOBAL_SETTINGS, STRATEGY_CONFIGS


load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")


async def fetch_price_data():
    symbols = ["BTCUSDT", "ETHUSDT"]
    async with BinanceDataLoader(
        api_key=API_KEY, 
        api_secret=API_SECRET,
        symbols=symbols,
        testnet=False,
    ) as loader:
        price_data = await loader.load_historical_data()
        return price_data


async def main():
    price_data = await fetch_price_data()

    strategies = [
        SMACrossoverStrategy(
            price_data,
            **{
                **GLOBAL_SETTINGS,
                **STRATEGY_CONFIGS["SMACrossoverStrategy"]
            }
        ),
        VWAPReversionStrategy(
            price_data,
            **{ 
                **GLOBAL_SETTINGS,
                **STRATEGY_CONFIGS["VWAPReversionStrategy"]
            }
        ),
        MultiTimeframeMomentumStrategy(
            price_data,
            **{ 
                **GLOBAL_SETTINGS,
                **STRATEGY_CONFIGS["MultiTimeframeMomentumStrategy"]
            }
            
        )
    ]
    backtester = Backtester(strategies)
    backtester.run_backtest()


if __name__ == "__main__":
    asyncio.run(main())
