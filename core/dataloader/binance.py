from typing import List
import logging


from binance import AsyncClient
import pandas as pd


from core.dataloader.base import BaseDataLoader
from core.dataloader.exceptions import NetworkError, DataValidationError


class BinanceDataLoader(BaseDataLoader):
    """Async Binance data loader implementation."""
    KLINES_COLUMNS = (
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_base_vol", "taker_quote_vol", "ignore"
    )

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: List[str] = None,
        start_date: str = "1 Mar, 2025",
        end_date: str = "28 Mar, 2025",
        timeframe: str = "1m",
        base_asset: str = "BTC",
        pairs_limit: int = 100,
        testnet: bool = False,
        cache_dir: str = "data"
    ):
        super().__init__(symbols, start_date, end_date, timeframe, cache_dir)
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_asset = base_asset
        self.pairs_limit = pairs_limit
        self.testnet = testnet
        self.client = None
        self._init_binance_logger() 

    def _init_binance_logger(self):
        """Initialize Binance-specific logging."""
        self.binance_logger = logging.getLogger(f"{self.__class__.__name__}.Binance")
        self.binance_logger.setLevel(logging.DEBUG)

    async def __aenter__(self):
        """Initialize Binance client and symbols."""
        self.binance_logger.debug("Establishing Binance API connection")
        try:
            self.client = await AsyncClient.create(
                self.api_key,
                self.api_secret,
                testnet=self.testnet
            )
        except Exception as e:
            self.binance_logger.error(f"Connection failed: {str(e)}")
            raise NetworkError("API connection error") from e
        
        if not self.symbols:
            self.binance_logger.info(f"Fetching top {self.pairs_limit} {self.base_asset} pairs by liquidity")
            self.symbols = await self.get_top_liquid_pairs(
                self.base_asset,
                self.pairs_limit
            )
            self.binance_logger.info(f"Selected symbols: {self.symbols}")

        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Clean up resources."""
        if self.client:
            self.binance_logger.debug("Closing Binance API connection")
            await self.client.close_connection()
        if exc:
            self.binance_logger.error(f"Exception occurred: {exc}", exc_info=True)

    async def fetch_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Fetch OHLCV data from Binance API."""
        self.binance_logger.info(f"Fetching data for {symbol}")
        try:
            klines = await self.client.get_historical_klines(
                symbol, self.timeframe, self.start_date, self.end_date
            )

            if not klines:
                self.binance_logger.warning(f"No data for {symbol}")
                raise DataValidationError(f"No data received for {symbol}")
            
            df = self._process_klines(klines, symbol)
            return df
        
        except DataValidationError:
            raise
        except Exception as e:
            self.binance_logger.error(f"API request failed for {symbol}: {e}")
            raise NetworkError(f"API request failed for {symbol}: {e}")
        

    def _process_klines(self, klines: List, symbol: str) -> pd.DataFrame:
        """Process raw klines into DataFrame."""
        self.binance_logger.debug(f"Processing {len(klines)} records for {symbol}")
        try:
            df = pd.DataFrame(klines, columns=self.KLINES_COLUMNS)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df.set_index("timestamp", inplace=True)
            
            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

            df = df.dropna()  # Drop any rows with NaN values
            if df.empty:
                raise DataValidationError("All data contained NaN values after cleaning", symbol)
            
            # Add symbol identifier
            df["symbol"] = symbol

            self.binance_logger.info(f"Successfully processed {symbol}")
            return df

        except Exception as e:
            self.binance_logger.error(f"Processing failed for {symbol}: {e}", exc_info=True)
            raise DataValidationError(f"Data processing error for {symbol}: {e}")

    async def get_top_liquid_pairs(self, base_asset: str, limit: int) -> List[str]:
        """Fetch top liquid pairs from Binance."""
        try:
            tickers = await self.client.get_ticker()
            relevant = [
                (t["symbol"], float(t["quoteVolume"]))
                for t in tickers if t["symbol"].endswith(base_asset)
            ]
            sorted_pairs = sorted(relevant, key=lambda x: x[1], reverse=True)
            pairs = [pair[0] for pair in sorted_pairs[:limit]]

            if not pairs:
                raise DataValidationError(f"No pairs found for {base_asset}")
            
            return pairs
        
        except Exception as e:
            self.binance_logger.error(f"Liquid pairs fetch failed: {e}")
            raise NetworkError(f"Failed to fetch liquid pairs: {e}")
