# Глобальні налаштування для всіх стратегій
GLOBAL_SETTINGS = {
    "fees": 0.001,          # 0.1% комісії (можна перевизначити для кожної стратегії)
    "slippage": 0.0005,     # 0.05% сліпейджу
    "init_cash": 10000,     # стартовий капітал $10,000
}

# Конфіги для кожної стратегії
STRATEGY_CONFIGS = {
    "SMACrossoverStrategy": {
        "short_window": 50,
        "long_window": 200,
        "fees": 0.001,
        "slippage": 0.001,
    },
    "VWAPReversionStrategy": {
        "lookback_window": 50,
        "deviation_threshold": 1.5,
        "fees": 0.002,
        "slippage": 0.001,
    },
    "MultiTimeframeMomentumStrategy": {
        "fast_period": 14,
        "slow_period": 50,
        "timeframes": ["5min", "15min"],
        "fees": 0.001,
        "slippage": 0.0005,
    }
}