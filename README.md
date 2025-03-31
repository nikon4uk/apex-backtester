# Trading Strategies Backtester

## Опис проекту

Цей проект містить набір торгових стратегій для криптовалютного ринку, реалізованих з використанням Python та бібліотеки vectorbt для бектестингу. Наразі проект підтримує три різні стратегії та дозволяє проводити їх тестування на історичних даних з Binance.

## Встановлення та запуск

1. **Встановлення залежностей**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Для Linux/Mac
   # або
   venv\Scripts\activate    # Для Windows
   pip install -r requirements.txt
   ```

2. **Налаштування API ключів Binance(ОПЦІОНАЛЬНО)**:
    *Я вже згенерував ключі для тестового акаунту та завантажив їх у проект*
   - Створіть файл `.env` в корені проекту
   - Додайте ваші API ключі:
     ```
     BINANCE_API_KEY=ваш_api_key
     BINANCE_API_SECRET=ваш_api_secret
     ```
   

3. **Запуск бектесту**:
   ```bash
   python main.py
   ```
# Архітектура програми

Програма складається з кількох основних модулів, які взаємодіють між собою для завантаження даних, виконання стратегій та аналізу результатів.

## Основні компоненти

### 1. Data Loading Layer
- **base.py** - Базовий абстрактний клас для завантажувачів даних
- **binance.py** - Реалізація для завантаження даних з Binance API
- **exceptions.py** - Власні винятки для обробки помилок

**Функціонал**:
- Асинхронне завантаження OHLCV даних
- Кешування даних у форматі Parquet
- Валідація отриманих даних
- Автоматичний вибір ліквідних пар

### 2. Strategy Layer
- **base.py** (StrategyBase) - Базовий клас для всіх стратегій
- Конкретні реалізації стратегій (sma_cross.py, vwarp_reversion.py, multi_timeframe_momentum.py)

**Функціонал**:
- Генерація торгових сигналів
- Виконання бектесту за допомогою vectorbt
- Розрахунок метрик продуктивності

### 3. Backtesting Engine
- **backtester.py** - Основний клас для керування процесом бектесту

**Функціонал**:
- Координація виконання стратегій
- Збір та збереження результатів
- Генерація візуалізацій

### 4. Допоміжні модулі
- **config.py** - Централізоване зберігання параметрів
- **visualizer.py** - Візуалізація результатів
- **metrics.py** - Налаштування метрик продуктивності

## Потік даних

1. Завантаження даних через BinanceDataLoader
2. Передача даних у стратегії
3. Генерація сигналів кожною стратегією
4. Виконання бектесту через vectorbt
5. Аналіз результатів та генерація звітів

## Ключові особливості архітектури

- **Модульність**: Кожен компонент ізольований та може бути легко замінений
- **Асинхронність**: Використання asyncio для ефективного завантаження даних
- **Кешування**: Автоматичне зберігання даних для подальшого використання
- **Уніфікований інтерфейс**: Всі стратегії наслідують один базовий клас
- **Гнучкість**: Легке додавання нових стратегій та джерел даних

Ця архітектура дозволяє легко розширювати функціонал, додаючи нові стратегії або джерела даних, зберігаючи при цьому чітку структуру проекту.
## Опис стратегій

### 1. SMA Crossover Strategy

**Принцип роботи**:  
Стратегія, заснована на перетині двох ковзних середніх (SMA). Сигнали генеруються, коли короткострокова SMA перетинає довгострокову SMA знизу вгору (сигнал на купівлю) або зверху вниз (сигнал на продаж).

**Параметри (за замовчуванням)**:
- Коротке вікно SMA: 50 періодів
- Довге вікно SMA: 200 періодів
- Комісія: 0.1%
- Сліппейдж: 0.1%

### 2. VWAP Reversion Strategy

**Принцип роботи**:  
Стратегія, яка використовує відхилення ціни від VWAP (Volume Weighted Average Price) для визначення точок входу. Стратегія передбачає купівлю, коли ціна значно нижче VWAP, і продаж, коли ціна значно вище VWAP.

**Параметри (за замовчуванням)**:
- Вікно для VWAP: 50 періодів
- Поріг відхилення: 1.5 стандартних відхилень
- Множник Take-Profit: 1.0
- Множник Stop-Loss: 1.5
- Комісія: 0.2%
- Сліппейдж: 0.1%

### 3. Multi-Timeframe Momentum Strategy

**Принцип роботи**:  
Стратегія, яка аналізує тренд на кількох таймфреймах одночасно. Використовує комбінацію сигналів від різних таймфреймів для підтвердження тренду.

**Параметри (за замовчуванням)**:
- Швидкий період: 14 періодів
- Повільний період: 50 періодів
- Таймфрейми: 5 хвилин, 15 хвилин
- Комісія: 0.1%
- Сліппейдж: 0.05%

## Налаштування стратегій

Всі параметри стратегій можна змінити у файлі `config.py`. Файл містить:
- Глобальні налаштування для всіх стратегій (`GLOBAL_SETTINGS`)
- Індивідуальні налаштування для кожної стратегії (`STRATEGY_CONFIGS`)

## Висновки

*Тут будуть додані висновки після аналізу результатів бектесту*