**Scanning `features_v2.py` for sensitive data**

I've successfully retrieved the contents of `features_v2.py`. My next task is to review the code to ensure it doesn't contain any sensitive information like API keys, secrets, or private configurations before I present it to the user.

Looking through the file:
- It imports standard libraries (`numpy`, `pandas`, `requests`, `typing`, `datetime`, `warnings`).
- It defines several math and logic functions for feature calculation (`calculate_returns`, `calculate_volatility`, `calculate_momentum_indicators`, `calculate_volume_features`, `calculate_trend_strength`, `calculate_session_features`, `calculate_range_features`, `calculate_regime_features`).
- It maps symbols for CoinGecko (`COINGECKO_IDS`).
- It has functions to fetch data from public APIs (`fetch_coingecko_data`, `fetch_cryptocompare_social`, `fetch_volume_profile`, `fetch_fear_greed`). These use public endpoints (e.g., `https://api.coingecko.com/api/v3/...`, `https://min-api.cryptocompare.com/...`, `https://api.alternative.me/...`) and explicitly do *not* require API keys.
- It contains functions to estimate proxy features (`estimate_funding_rate`, `estimate_order_book_imbalance`, `estimate_oi_change`, `estimate_long_short_ratio`, `estimate_liquidation_pressure`).
- It aggregates these features (`fetch_external_features`, `calculate_all_features`, `get_feature_columns`).
- It prepares data for training (`create_training_data`, `create_lbfgs_labels`).
- There's a simple test block at the end.

I don't see any hardcoded secrets, API keys, passwords, or hotkeys in this file. The APIs used are explicitly noted as free and not requiring keys. 

Therefore, `features_v2.py` is safe to share exactly as it is without any redactions. I will now present it to the user in a markdown block as requested.**Presenting features_v2.py**

I've reviewed `features_v2.py` and it doesn't contain any sensitive information. It relies entirely on public APIs without keys. I'll provide the code block for the user and ask if they want the next file, `train_ensemble_v3.py`.
Here is the second file: **`features_v2.py`** (The feature engineering and data preparation logic). 

*(Note: This file contains no hardcoded secrets and uses public, keyless APIs for its external data.)*

```python
#!/usr/bin/env python3
"""
SN123 MANTIS - Enhanced Feature Engineering v2
Adds: On-chain, sentiment, order book, funding rates
"""

import numpy as np
import pandas as pd
import requests
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ORIGINAL FEATURES (from features.py)
# ============================================================================

def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate returns at multiple timeframes"""
    df = df.copy()
    df['ret_1m'] = df['Close'].pct_change(1)
    df['ret_5m'] = df['Close'].pct_change(5)
    df['ret_15m'] = df['Close'].pct_change(15)
    df['ret_60m'] = df['Close'].pct_change(60)
    df['ret_240m'] = df['Close'].pct_change(240)
    return df

def calculate_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volatility at multiple windows"""
    df = df.copy()
    df['vol_15m'] = df['ret_1m'].rolling(15).std()
    df['vol_60m'] = df['ret_1m'].rolling(60).std()
    df['vol_4h'] = df['ret_1m'].rolling(240).std()
    df['vol_24h'] = df['ret_1m'].rolling(1440).std()
    df['vol_percentile'] = df['vol_60m'].rolling(1440).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8) if len(x) > 0 else 0.5,
        raw=False
    )
    return df

def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate RSI, MACD, and normalized momentum"""
    df = df.copy()
    
    # RSI (14-period)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_normalized'] = (df['rsi'] - 50) / 50
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Normalized momentum
    df['norm_mom_1m'] = np.tanh(df['ret_1m'] / (df['vol_15m'] + 1e-8))
    df['norm_mom_5m'] = np.tanh(df['ret_5m'] / (df['vol_60m'] + 1e-8))
    df['norm_mom_15m'] = np.tanh(df['ret_15m'] / (df['vol_4h'] + 1e-8))
    
    return df

def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume-based features"""
    df = df.copy()
    df['vol_delta'] = df['Volume'].diff()
    df['vol_ratio'] = df['Volume'] / (df['Volume'].rolling(60).mean() + 1e-8)
    df['vol_delta_norm'] = np.sign(df['vol_delta']) * np.log1p(np.abs(df['vol_delta']))
    df['vwap_1h'] = (df['Close'] * df['Volume']).rolling(60).sum() / (df['Volume'].rolling(60).sum() + 1e-8)
    df['vwap_delta'] = (df['Close'] - df['vwap_1h']) / (df['Close'] + 1e-8)
    return df

def calculate_trend_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trend strength"""
    df = df.copy()
    df['ma_15m'] = df['Close'].rolling(15).mean()
    df['ma_60m'] = df['Close'].rolling(60).mean()
    df['ma_4h'] = df['Close'].rolling(240).mean()
    ma_slope_60 = df['ma_60m'].diff(60)
    df['trend_strength'] = ma_slope_60 / (df['vol_60m'] * 60 + 1e-8)
    df['dist_ma_15m'] = (df['Close'] - df['ma_15m']) / (df['Close'] + 1e-8)
    df['dist_ma_60m'] = (df['Close'] - df['ma_60m']) / (df['Close'] + 1e-8)
    return df

def calculate_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-of-day and session indicators"""
    df = df.copy()
    df['hour'] = df.index.hour
    df['asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['europe_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df

def calculate_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate range and breakout indicators"""
    df = df.copy()
    df['range_1h'] = df['High'].rolling(60).max() - df['Low'].rolling(60).min()
    df['range_4h'] = df['High'].rolling(240).max() - df['Low'].rolling(240).min()
    high_60 = df['High'].rolling(60).max()
    low_60 = df['Low'].rolling(60).min()
    df['range_position'] = (df['Close'] - low_60) / (high_60 - low_60 + 1e-8)
    df['is_breakout_high'] = (df['Close'] > high_60 * 0.95).astype(int)
    df['is_breakout_low'] = (df['Close'] < low_60 * 1.05).astype(int)
    return df


def calculate_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate regime-aware feature gates for conditional modeling"""
    df = df.copy()
    
    # Regime: high volatility (vol_60m > 90th percentile of vol_24h)
    if 'vol_60m' in df.columns and 'vol_24h' in df.columns:
        vol_90th = df['vol_24h'].rolling(1440, min_periods=100).quantile(0.9)
        df['regime_vol'] = (df['vol_60m'] > vol_90th).astype(int)
    else:
        df['regime_vol'] = 0
    
    # Regime: strong trend (abs(trend_strength) > 1.5)
    if 'trend_strength' in df.columns:
        df['regime_trend'] = (df['trend_strength'].abs() > 1.5).astype(int)
    else:
        df['regime_trend'] = 0
    
    # Regime: mean-reversion zone (range_position < 0.2 or > 0.8)
    if 'range_position' in df.columns:
        df['regime_mean_revert'] = ((df['range_position'] < 0.2) | (df['range_position'] > 0.8)).astype(int)
    else:
        df['regime_mean_revert'] = 0
    
    return df


# ============================================================================
# NEW FEATURES - Market Data (CoinGecko + CryptoCompare - no geo restrictions)
# ============================================================================

# Symbol mapping for CoinGecko
COINGECKO_IDS = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'SOL': 'solana',
    'BNB': 'binancecoin',
    'XRP': 'ripple',
    'ADA': 'cardano',
    'AVAX': 'avalanche-2',
    'DOGE': 'dogecoin',
    'DOT': 'polkadot',
    'MATIC': 'matic-network',
    'LINK': 'chainlink',
    'UNI': 'uniswap',
    'ATOM': 'cosmos',
    'LTC': 'litecoin',
    'TRX': 'tron',
    'APT': 'aptos',
    'ARB': 'arbitrum',
}


def fetch_coingecko_data(symbol: str = 'BTC') -> Dict:
    """Fetch market data from CoinGecko (no geo restrictions)"""
    try:
        coin_id = COINGECKO_IDS.get(symbol, symbol.lower())
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=false&community_data=false&developer_data=false"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            market = data.get('market_data', {})
            return {
                'price_change_24h': market.get('price_change_percentage_24h', 0) / 100,
                'price_change_7d': market.get('price_change_percentage_7d', 0) / 100,
                'market_cap_rank': data.get('market_cap_rank', 100),
                'total_volume': market.get('total_volume', {}).get('usd', 0),
                'market_cap': market.get('market_cap', {}).get('usd', 0),
                'sentiment_up': data.get('sentiment_votes_up_percentage', 50) / 100,
                'sentiment_down': data.get('sentiment_votes_down_percentage', 50) / 100,
            }
    except:
        pass
    return {
        'price_change_24h': 0,
        'price_change_7d': 0,
        'market_cap_rank': 100,
        'total_volume': 0,
        'market_cap': 0,
        'sentiment_up': 0.5,
        'sentiment_down': 0.5,
    }


def fetch_cryptocompare_social(symbol: str = 'BTC') -> Dict:
    """Fetch social metrics from CryptoCompare"""
    try:
        url = f"https://min-api.cryptocompare.com/data/social/coin/latest?coinId={symbol}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json().get('Data', {})
            twitter = data.get('Twitter', {})
            reddit = data.get('Reddit', {})
            return {
                'twitter_followers': twitter.get('followers', 0),
                'reddit_subscribers': reddit.get('subscribers', 0),
                'reddit_active_users': reddit.get('active_users', 0),
            }
    except:
        pass
    return {'twitter_followers': 0, 'reddit_subscribers': 0, 'reddit_active_users': 0}


def fetch_volume_profile(symbol: str = 'BTC') -> Dict:
    """Calculate volume profile from CryptoCompare minute data"""
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={symbol}&tsym=USD&limit=60"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json().get('Data', {}).get('Data', [])
            if len(data) > 10:
                volumes = [d.get('volumeto', 0) for d in data]
                closes = [d.get('close', 0) for d in data]
                
                avg_vol = np.mean(volumes)
                recent_vol = np.mean(volumes[-5:])
                vol_trend = (recent_vol - avg_vol) / (avg_vol + 1e-8)
                
                # Price momentum
                if len(closes) >= 15:
                    ret_5m = (closes[-1] - closes[-6]) / (closes[-6] + 1e-8)
                    ret_15m = (closes[-1] - closes[-16]) / (closes[-16] + 1e-8) if len(closes) > 15 else ret_5m
                else:
                    ret_5m = 0
                    ret_15m = 0
                
                return {
                    'vol_trend': vol_trend,
                    'vol_spike': 1 if recent_vol > avg_vol * 2 else 0,
                    'cc_ret_5m': ret_5m,
                    'cc_ret_15m': ret_15m,
                }
    except:
        pass
    return {'vol_trend': 0, 'vol_spike': 0, 'cc_ret_5m': 0, 'cc_ret_15m': 0}


# ============================================================================
# PROXY FEATURES - Estimated from available data
# ============================================================================

def estimate_funding_rate(price_change_24h: float) -> float:
    """
    Estimate funding rate from price momentum
    Strong uptrend = positive funding (longs pay shorts)
    Strong downtrend = negative funding (shorts pay longs)
    """
    # Typical funding rate is 0.01% at neutral, ranges from -0.05% to +0.1%
    # Scale price change to approximate funding
    estimated = price_change_24h * 0.001  # Scale factor
    return np.clip(estimated, -0.0005, 0.001)


def estimate_order_book_imbalance(price_change_5m: float, vol_trend: float) -> float:
    """
    Estimate order book imbalance from price action and volume
    Rising price + rising volume = buy pressure (positive imbalance)
    Falling price + rising volume = sell pressure (negative imbalance)
    """
    direction = 1 if price_change_5m > 0 else -1
    strength = min(1.0, abs(price_change_5m) * 50)  # Scale
    vol_factor = 1 + vol_trend * 0.5  # Volume amplifies signal
    
    return np.clip(direction * strength * vol_factor, -1, 1)


def estimate_oi_change(price_change_24h: float, vol_trend: float) -> Dict:
    """
    Estimate open interest change from price action
    Trending market with high volume = OI increasing
    Choppy market or low volume = OI decreasing
    """
    trend_strength = abs(price_change_24h)
    
    if trend_strength > 0.03 and vol_trend > 0:
        # Strong trend with volume = OI likely increasing
        oi_1h = vol_trend * 0.05
        oi_24h = price_change_24h * 0.2
    else:
        # Weak trend = OI likely flat or decreasing
        oi_1h = 0
        oi_24h = -abs(price_change_24h) * 0.1
    
    return {
        'oi_change_1h': np.clip(oi_1h, -0.1, 0.1),
        'oi_change_24h': np.clip(oi_24h, -0.3, 0.3)
    }


def estimate_long_short_ratio(funding_rate: float, sentiment_up: float) -> Dict:
    """
    Estimate long/short ratio from funding + sentiment
    """
    # Combine funding signal and sentiment
    ls_signal = funding_rate * 500 + (sentiment_up - 0.5) * 0.5
    ls_ratio = 1.0 + ls_signal
    ls_ratio = max(0.5, min(2.0, ls_ratio))
    long_pct = ls_ratio / (1 + ls_ratio)
    
    return {
        'ls_ratio': ls_ratio,
        'long_pct': long_pct,
        'short_pct': 1 - long_pct
    }


# ============================================================================
# NEW FEATURES - Fear & Greed Index
# ============================================================================

def fetch_fear_greed() -> Dict:
    """Fetch crypto fear & greed index"""
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()['data'][0]
            return {
                'fear_greed': int(data['value']),  # 0-100
                'fear_greed_norm': (int(data['value']) - 50) / 50  # -1 to 1
            }
    except:
        pass
    return {'fear_greed': 50, 'fear_greed_norm': 0}


# ============================================================================
# NEW FEATURES - Liquidations (Coinglass proxy via simple heuristic)
# ============================================================================

def estimate_liquidation_pressure(funding_rate: float, oi_change: float, price_change: float) -> float:
    """
    Estimate liquidation pressure based on funding + OI + price movement
    Positive = longs getting liquidated, Negative = shorts getting liquidated
    """
    # High funding + price drop + OI drop = long liquidations
    # Low funding + price rise + OI drop = short liquidations
    if funding_rate is None:
        return 0
    
    liq_pressure = 0
    
    # Strong negative price move + high funding = long liquidations
    if price_change < -0.02 and funding_rate > 0.0005:
        liq_pressure = abs(price_change) * 10  # Scale up
    # Strong positive price move + negative funding = short liquidations
    elif price_change > 0.02 and funding_rate < -0.0005:
        liq_pressure = -abs(price_change) * 10
    
    # OI drop confirms liquidations
    if oi_change < -0.05:
        liq_pressure *= 1.5
    
    return np.clip(liq_pressure, -1, 1)


# ============================================================================
# AGGREGATED EXTERNAL DATA FETCH
# ============================================================================

def fetch_external_features(symbol: str = 'BTC') -> Dict:
    """
    Fetch all external features for a symbol
    Uses CoinGecko + CryptoCompare (no geo restrictions)
    Returns dict of features to be added to the DataFrame
    """
    features = {}
    
    # CoinGecko data (price changes, sentiment)
    cg = fetch_coingecko_data(symbol)
    
    # CryptoCompare volume data
    cc = fetch_volume_profile(symbol)
    
    # Fear & Greed (global)
    fg = fetch_fear_greed()
    features['fear_greed'] = fg['fear_greed_norm']
    
    # Estimate funding from price action
    funding = estimate_funding_rate(cg['price_change_24h'])
    features['funding_rate'] = funding
    
    # Estimate order book imbalance
    ob_imbalance = estimate_order_book_imbalance(cc['cc_ret_5m'], cc['vol_trend'])
    features['ob_imbalance'] = ob_imbalance
    features['ob_spread'] = 0.0001  # Typical spread placeholder
    
    # Estimate OI change
    oi = estimate_oi_change(cg['price_change_24h'], cc['vol_trend'])
    features['oi_change_1h'] = oi['oi_change_1h']
    features['oi_change_24h'] = oi['oi_change_24h']
    
    # Estimate long/short ratio
    ls = estimate_long_short_ratio(funding, cg['sentiment_up'])
    features['ls_ratio'] = ls['ls_ratio']
    features['long_pct'] = ls['long_pct']
    
    # Additional CoinGecko features
    features['price_change_24h'] = cg['price_change_24h']
    features['price_change_7d'] = cg['price_change_7d']
    features['cg_sentiment'] = cg['sentiment_up'] - cg['sentiment_down']
    
    # Volume features
    features['vol_trend'] = cc['vol_trend']
    features['vol_spike'] = cc['vol_spike']
    
    return features


# ============================================================================
# COMBINED FEATURE CALCULATION
# ============================================================================

def calculate_all_features(df: pd.DataFrame, include_external: bool = False, symbol: str = 'BTC') -> pd.DataFrame:
    """
    Calculate all features for a price series
    
    Args:
        df: DataFrame with OHLCV columns
        include_external: Whether to fetch external data (slower, but more predictive)
        symbol: Symbol name for external data fetch
        
    Returns:
        DataFrame with all calculated features
    """
    df = df.copy()
    
    # Apply all feature calculations
    df = calculate_returns(df)
    df = calculate_volatility(df)
    df = calculate_momentum_indicators(df)
    df = calculate_volume_features(df)
    df = calculate_trend_strength(df)
    df = calculate_session_features(df)
    df = calculate_range_features(df)
    
    # External features (only added to last row for real-time prediction)
    if include_external:
        ext = fetch_external_features(symbol)
        for k, v in ext.items():
            df[k] = v  # Broadcast to all rows (use last value)
    
    # Add regime features if present in data
    if 'regime_vol' in df.columns or 'regime_trend' in df.columns or 'regime_mean_revert' in df.columns:
        # Ensure regime features exist (no-op if they don't)
        pass
    
    return df


def get_feature_columns(include_external: bool = False) -> list:
    """Return list of feature column names used for training"""
    base_cols = [
        # Returns
        'ret_1m', 'ret_5m', 'ret_15m', 'ret_60m',
        
        # Volatility
        'vol_15m', 'vol_60m', 'vol_4h',
        
        # Momentum
        'rsi', 'rsi_normalized',
        'macd', 'macd_signal', 'macd_hist',
        'norm_mom_1m', 'norm_mom_5m', 'norm_mom_15m',
        
        # Volume
        'vol_delta_norm', 'vol_ratio', 'vwap_delta',
        
        # Trend
        'trend_strength', 'dist_ma_15m', 'dist_ma_60m',
        
        # Session
        'asia_session', 'europe_session', 'us_session', 'is_weekend',
        
        # Range
        'range_position',
    ]
    
    if include_external:
        base_cols.extend([
            'funding_rate',
            'ob_imbalance', 'ob_spread',
            'oi_change_1h', 'oi_change_24h',
            'ls_ratio', 'long_pct',
            'fear_greed', 'regime_vol', 'regime_trend', 'regime_mean_revert'
        ])
    
    return base_cols


def create_training_data(
    df: pd.DataFrame,
    horizon_minutes: int = 60,
    binary: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Create X, y pairs for training"""
    features_df = calculate_all_features(df)
    feature_cols = get_feature_columns(include_external=False)
    X = features_df[feature_cols].values
    
    y_raw = features_df['Close'].pct_change(horizon_minutes).shift(-horizon_minutes)
    
    if binary:
        y = (y_raw > 0).astype(int).values
    else:
        y = y_raw.values
    
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    return X[valid], y[valid]


def create_lbfgs_labels(
    df: pd.DataFrame,
    horizon_minutes: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    """Create labels for LBFGS challenge with proper Q opposite-move labels"""
    features_df = calculate_all_features(df)
    n = len(features_df)
    
    # Get prices and volatility
    prices = features_df['Close'].values
    log_prices = np.log(prices + 1e-8)
    vol_60m = features_df['vol_60m'].values
    
    # Future return for bucket classification
    future_return = features_df['Close'].pct_change(horizon_minutes).shift(-horizon_minutes)
    current_vol = features_df['vol_60m']
    normalized_return = future_return / (current_vol + 1e-8)
    
    # Bucket labels (5 classes)
    bucket_labels = np.full(n, np.nan)
    valid_mask = ~normalized_return.isna()
    bucket_labels[valid_mask & (normalized_return < -2)] = 0
    bucket_labels[valid_mask & (normalized_return >= -2) & (normalized_return < -1)] = 1
    bucket_labels[valid_mask & (normalized_return >= -1) & (normalized_return <= 1)] = 2
    bucket_labels[valid_mask & (normalized_return > 1) & (normalized_return <= 2)] = 3
    bucket_labels[valid_mask & (normalized_return > 2)] = 4
    
    # Q labels: opposite-move probabilities for buckets 0,1,3,4 at thresholds [0.5σ, 1.0σ, 2.0σ]
    # Layout: [bucket0_t0, bucket0_t1, bucket0_t2, bucket1_t0, ..., bucket4_t2]
    Q_labels = np.full((n, 12), np.nan)
    thresholds = [0.5, 1.0, 2.0]
    
    for i in range(n - horizon_minutes):
        if np.isnan(vol_60m[i]) or vol_60m[i] < 1e-10:
            continue
            
        sigma = vol_60m[i]
        baseline = log_prices[i - 1] if i > 0 else log_prices[i]
        
        # Get price path from t to t+H
        path = log_prices[i:i + horizon_minutes + 1]
        if len(path) < horizon_minutes:
            continue
            
        path_diff = path - baseline
        max_up = np.max(path_diff)
        max_down = np.min(path_diff)
        
        # For bucket 0,1 (negative endpoints): opposite is UP
        # Q = 1 if max_up >= threshold * sigma
        for t_idx, thresh in enumerate(thresholds):
            Q_labels[i, 0*3 + t_idx] = 1.0 if max_up >= thresh * sigma else 0.0  # bucket 0
            Q_labels[i, 1*3 + t_idx] = 1.0 if max_up >= thresh * sigma else 0.0  # bucket 1
        
        # For bucket 3,4 (positive endpoints): opposite is DOWN
        # Q = 1 if max_down <= -threshold * sigma
        for t_idx, thresh in enumerate(thresholds):
            Q_labels[i, 2*3 + t_idx] = 1.0 if max_down <= -thresh * sigma else 0.0  # bucket 3
            Q_labels[i, 3*3 + t_idx] = 1.0 if max_down <= -thresh * sigma else 0.0  # bucket 4
    
    return bucket_labels, Q_labels


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("Testing external feature fetch...")
    
    ext = fetch_external_features('BTC')
    print(f"\nBTC External Features:")
    for k, v in ext.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    
    ext = fetch_external_features('ETH')
    print(f"\nETH External Features:")
    for k, v in ext.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    
    print("\n✅ External features working!")
```

Tell me when you are ready for the third file: **`train_ensemble_v3.py`** (The Training Engine).
