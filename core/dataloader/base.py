import os
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


import pandas as pd


from core.dataloader.exceptions import CacheError, DataLoaderError, DataValidationError, NetworkError


logger = logging.getLogger("BaseDataLoader")
logger.setLevel(logging.DEBUG)


class BaseDataLoader(ABC):
    """Abstract base class for market data loaders."""

    def __init__(
        self,
        symbols: List[str] = None,
        start_date: str = "",
        end_date: str = "",
        timeframe: str = "",
        cache_dir: str = "data"
    ):  
        self._init_logger()
        self.symbols = symbols or []
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(f"Initialized base loader with {len(self.symbols)} symbols")

    def _init_logger(self):
        """Initialize base logger instance."""
        self.logger = logging.getLogger(f"{self.__class__.__name__}.Base")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    @abstractmethod
    async def fetch_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Fetch OHLCV data for a single symbol."""
        pass

    @abstractmethod
    async def get_top_liquid_pairs(self, base_asset: str = 'BTC', limit: int = 100) -> List[str]:
        """Get top liquid pairs for base asset."""
        pass

    async def load_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Load data for all symbols in parallel with caching."""
        self.logger.info(f"Starting data load for {len(self.symbols)} symbols")

        if not self.symbols:
            self.logger.warning("No symbols to load")
            raise DataLoaderError("No symbols configured")
        
        tasks = {
            symbol: asyncio.create_task(
                self._load_symbol_data(symbol),
                name=f"load_{symbol}"
                )
            for symbol in self.symbols
        }

        data = {}
        # Обробляємо результати паралельно
        for symbol, task in tasks.items():
            try:
                data[symbol] = await task
                self.logger.debug(f"Success: {symbol}")
            except DataLoaderError as e:
                self.logger.warning(f"Skipped {symbol}: {e}")
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error with {symbol}: {e}", exc_info=True)
                
        if not data:
            self.logger.error("No data loaded for any symbol")
            raise DataLoaderError("No valid data loaded for any symbol")
        
        self.logger.info(f"Completed: {len(data)}/{len(self.symbols)} loaded")
        return data

    async def _load_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Load data for a single symbol with caching."""
        self.logger.debug(f"Processing pipeline for {symbol}")
        filename = self._generate_cache_filename(symbol)
        
        try:
            cached_data = self._load_from_cache(filename)
            if cached_data is not None:
                self._validate_data(cached_data, symbol)
                self.logger.info(f"Cache hit: {symbol}")
                return cached_data
        except CacheError as e:
            self.logger.warning(f"Cache read failed for {symbol}: {e}")

        # Fetch fresh data if cache miss or invalid
        fresh_data = await self.fetch_ohlcv(symbol)
        self._validate_data(fresh_data, symbol)

        try:
            self._save_to_cache(fresh_data, filename)
            self.logger.debug(f"Cached: {symbol}")
        except CacheError as e:
            self.logger.warning(f"Cache write failed for {symbol}: {e}")

        return fresh_data

    def _generate_cache_filename(self, symbol: str) -> str:
        """Generate cache filename for symbol."""
        return f"{symbol}_{self.timeframe}_{self.start_date}_{self.end_date}"

    def _save_to_cache(self, data: pd.DataFrame, filename: str) -> None:
        """Save data to cache."""
        try:
            filepath = os.path.join(self.cache_dir, f"{filename}.parquet")
            data.to_parquet(filepath, compression="gzip")
        except Exception as e:
            raise CacheError(f"Failed to save cache: {e}")

    def _load_from_cache(self, filename: str) -> Optional[pd.DataFrame]:
        """Load data from cache."""
        filepath = os.path.join(self.cache_dir, f"{filename}.parquet")
        if os.path.exists(filepath):
            try:
                return pd.read_parquet(filepath)
            except Exception as e:
                raise CacheError(f"Failed to load cache: {e}")
        return None

    def _validate_data(self, df: pd.DataFrame, symbol: str) -> None:
        """Validate DataFrame structure and content."""
        required_columns = {"open", "high", "low", "close", "volume"}
        
        if df.empty:
            raise DataValidationError(f"Empty DataFrame for {symbol}")
            
        if not required_columns.issubset(df.columns):
            raise DataValidationError(f"Missing required columns in {symbol}")
            
        if df.isna().any().any():
            raise DataValidationError(f"NaN values found in {symbol}")
            
        if (df[list(required_columns)] < 0).any().any():
            raise DataValidationError(f"Negative values in {symbol}")
            
        if df.index.duplicated().any():
            raise DataValidationError(f"Duplicate timestamps in {symbol}")
        
        if not all(df[col].dtype in [float, int] for col in required_columns):
            raise DataValidationError("Incorrect data format in price or volume columns", symbol)