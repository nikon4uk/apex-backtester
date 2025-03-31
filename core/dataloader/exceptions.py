class DataLoaderError(Exception):
    """Base exception for all data loader errors."""
    pass


class NetworkError(DataLoaderError):
    """Raised when there's a network-related error."""
    pass


class DataValidationError(DataLoaderError):
    """Raised when data validation fails."""
    pass


class CacheError(DataLoaderError):
    """Raised when there's an issue with caching."""
    pass
