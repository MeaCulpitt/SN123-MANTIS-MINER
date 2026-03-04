**Scanning `train_ensemble_v3.py` for sensitive data**

I've successfully retrieved the contents of `train_ensemble_v3.py`. Now, I need to review the code to ensure it doesn't contain any sensitive information like API keys, secrets, or private configurations before I present it to the user.

Looking through the file:
- It imports standard libraries for machine learning (`numpy`, `pandas`, `xgboost`, `sklearn`, `optuna`).
- It defines helper classes (`DummyModel`, `TemperatureScaler`).
- It defines the core ensemble model class (`EnhancedEnsembleClassifier`) which handles hyperparameter tuning with Optuna, training XGBoost, deep MLPs, and Gradient Boosting models, performing calibration, and temperature scaling.
- It contains functions for adding features (`add_cross_asset_features`, `add_seasonality_features`).
- It has functions that define the training pipeline for different model types (`load_data`, `train_binary_model`, `train_breakout_model`, `train_hitfirst_model`, `train_lbfgs_model`).
- Finally, it has the `__main__` execution block that orchestrates training across all symbols (Binary, Forex, HITFIRST, LBFGS, Multi-Breakout) and saves the serialized models (`.pkl`).

I do not see any hardcoded secrets, API keys, passwords, bucket names, endpoint URLs, or hotkeys in this file. It is pure machine learning structure and training logic.

Therefore, `train_ensemble_v3.py` is safe to share exactly as it is without any redactions. I will now present it to the user in a markdown block as requested, and note that it's the final file of the three main ones discussed.**Presenting train_ensemble_v3.py**

I've reviewed `train_ensemble_v3.py` and it doesn't contain any sensitive information. It's solely the logic for training the ML ensemble. I'll provide the code block for the user. I should also ask if there are any other specific files they want, like `cycle.py` or `.env` template.
Here is the third file: **`train_ensemble_v3.py`** (The Training Engine).

*(Note: This file contains the machine learning architecture and training pipeline logic. It contains no hardcoded secrets.)*

```python
#!/usr/bin/env python3
"""
SN123 MANTIS - Ensemble Training v3
OPTIMIZATIONS:
1. Hyperparameter tuning with Optuna (enabled)
2. Cross-asset correlation features
3. Time-of-day seasonality
4. Temperature-scaled probability calibration
5. Larger neural networks
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# Hyperparameter tuning
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from features_v2 import calculate_all_features, get_feature_columns, create_training_data, create_lbfgs_labels

MODEL_DIR = Path(__file__).parent / 'models'
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = get_feature_columns(include_external=False)


class DummyModel:
    """Fallback model for edge cases"""
    def __init__(self, n_classes=2):
        self.n_classes = n_classes
    
    def predict_proba(self, X):
        return np.ones((len(X), self.n_classes)) / self.n_classes
    
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class TemperatureScaler:
    """Post-hoc temperature scaling for better calibration"""
    
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, logits, labels):
        """Find optimal temperature via grid search"""
        from scipy.optimize import minimize_scalar
        
        def nll(T):
            scaled = logits / T
            probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
            # Cross entropy
            return -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-10))
        
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        return self
    
    def transform(self, probs):
        """Apply temperature scaling"""
        if self.temperature == 1.0:
            return probs
        # Convert to logits, scale, convert back
        logits = np.log(probs + 1e-10)
        scaled = logits / self.temperature
        return np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)


class EnhancedEnsembleClassifier:
    """
    V3 Ensemble with:
    - XGBoost (tuned)
    - Gradient Boosting
    - Deep MLP
    - Temperature-scaled calibration
    """
    
    def __init__(self, n_classes=2):
        self.n_classes = n_classes
        self.scaler = RobustScaler()  # More robust to outliers
        self.temp_scaler = TemperatureScaler()
        self.models = {}
        self.weights = {}
        self.meta_model = None
        self.is_fitted = False
    
    def _tune_xgb(self, X, y, n_trials=15):
        """Tune XGBoost with Optuna"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'max_bin': trial.suggest_int('max_bin', 128, 512),
            }
            
            model = XGBClassifier(
                **params,
                random_state=42,
                eval_metric='logloss',
                tree_method='hist',
                n_jobs=-1,
                objective='multi:softprob' if self.n_classes > 2 else 'binary:logistic',
                num_class=self.n_classes if self.n_classes > 2 else None
            )
            
            
            # Time-series CV
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_val)
                
                # Log loss
                from sklearn.metrics import log_loss
                scores.append(log_loss(y_val, probs))
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        return study.best_params
    
    def _create_xgb(self, params=None):
        default = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'n_jobs': -1
        }
        if params:
            default.update(params)
        
        if self.n_classes > 2:
            default['objective'] = 'multi:softprob'
            default['num_class'] = self.n_classes
        
        return XGBClassifier(**default)
    
    def _create_gb(self, params=None):
        """Gradient Boosting as secondary model"""
        default = {
            'n_estimators': 80,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'random_state': 42
        }
        if params:
            default.update(params)
        return GradientBoostingClassifier(**default)
    
    def _create_deep_mlp(self):
        """Deeper neural network"""
        return MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=64,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42
        )
    
    def _best_calibration(self, base_model, X_train, y_train, model_name):
        """Try both isotonic and sigmoid calibration, return whichever has lower val log-loss"""
        from sklearn.metrics import log_loss
        
        best_cal = None
        best_ll = float('inf')
        best_method = 'isotonic'
        
        for method in ['isotonic', 'sigmoid']:
            try:
                cal = CalibratedClassifierCV(base_model, method=method, cv=3)
                cal.fit(X_train, y_train)
                # Quick eval on training data (CV already used)
                probs = cal.predict_proba(X_train)
                ll = log_loss(y_train, probs)
                if ll < best_ll:
                    best_ll = ll
                    best_cal = cal
                    best_method = method
            except Exception:
                continue
        
        if best_cal is None:
            # Fallback: no calibration wrapper
            best_cal = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
            best_cal.fit(X_train, y_train)
            best_method = 'isotonic'
        
        return best_cal, best_method
    
    def fit(self, X, y, tune_hyperparams=True, n_trials=15):
        """Fit ensemble with hyperparameter tuning, calibration sweep, and stacking meta-model"""
        
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print("  ⚠️ Single class, using dummy")
            self.models['dummy'] = DummyModel(self.n_classes)
            self.is_fitted = True
            return self
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Tune and train XGBoost
        if tune_hyperparams and len(X) > 500:
            print("  Tuning XGBoost...", end=' ', flush=True)
            best_params = self._tune_xgb(X_train, y_train, n_trials=n_trials)
            print(f"done (depth={best_params.get('max_depth', 6)}, lr={best_params.get('learning_rate', 0.05):.3f})")
            xgb = self._create_xgb(best_params)
        else:
            xgb = self._create_xgb()
        
        # Calibrate XGBoost — sweep isotonic vs sigmoid
        print("  Training XGBoost...", end=' ', flush=True)
        xgb_calibrated, xgb_cal_method = self._best_calibration(xgb, X_train, y_train, 'xgb')
        self.models['xgb'] = xgb_calibrated
        print(f"done (cal={xgb_cal_method})")
        
        # Train deep MLP — sweep isotonic vs sigmoid
        print("  Training Deep MLP...", end=' ', flush=True)
        mlp = self._create_deep_mlp()
        try:
            mlp_calibrated, mlp_cal_method = self._best_calibration(mlp, X_train, y_train, 'mlp')
            self.models['mlp'] = mlp_calibrated
            print(f"done (cal={mlp_cal_method})")
        except Exception as e:
            print(f"failed ({e})")
            self.models['mlp'] = DummyModel(self.n_classes)
        
        # Train Gradient Boosting — sweep isotonic vs sigmoid
        print("  Training GradientBoosting...", end=' ', flush=True)
        gb = self._create_gb()
        try:
            gb_calibrated, gb_cal_method = self._best_calibration(gb, X_train, y_train, 'gb')
            self.models['gb'] = gb_calibrated
            print(f"done (cal={gb_cal_method})")
        except Exception as e:
            print(f"failed ({e})")
            self.models['gb'] = DummyModel(self.n_classes)
        
        # Calculate model weights based on validation performance
        print("  Calculating weights...", end=' ', flush=True)
        from sklearn.metrics import log_loss
        
        for name, model in self.models.items():
            probs = model.predict_proba(X_val)
            try:
                ll = log_loss(y_val, probs)
                # Convert log loss to weight (lower is better)
                self.weights[name] = 1.0 / (ll + 0.1)
            except:
                self.weights[name] = 1.0
        
        # Normalize weights
        total = sum(self.weights.values())
        for name in self.weights:
            self.weights[name] /= total
        
        print(f"XGB:{self.weights.get('xgb', 0):.2f} MLP:{self.weights.get('mlp', 0):.2f} GB:{self.weights.get('gb', 0):.2f}")
        
        # Train stacking meta-model (LogisticRegression on base model predictions)
        print("  Training meta-model...", end=' ', flush=True)
        try:
            meta_features_val = []
            for name in sorted(self.models.keys()):
                model = self.models[name]
                if isinstance(model, DummyModel):
                    continue
                preds = model.predict_proba(X_val)
                if preds.shape[1] == self.n_classes:
                    meta_features_val.append(preds)
            
            if len(meta_features_val) >= 2:
                meta_X = np.hstack(meta_features_val)
                self.meta_model = LogisticRegression(max_iter=1000)
                self.meta_model.fit(meta_X, y_val)
                meta_probs = self.meta_model.predict_proba(meta_X)
                meta_ll = log_loss(y_val, meta_probs)
                print(f"done (log-loss={meta_ll:.4f})")
            else:
                print("skipped (not enough base models)")
        except Exception as e:
            print(f"failed ({e})")
            self.meta_model = None
        
        # Temperature scaling on validation set
        print("  Temperature calibration...", end=' ', flush=True)
        ensemble_probs = self._predict_ensemble(X_val)
        try:
            self.temp_scaler.fit(np.log(ensemble_probs + 1e-10), y_val)
            print(f"T={self.temp_scaler.temperature:.2f}")
        except:
            print("skipped")
        
        self.is_fitted = True
        return self
    
    def _predict_ensemble(self, X):
        """Weighted ensemble prediction"""
        probs = np.zeros((len(X), self.n_classes))
        
        for name, model in self.models.items():
            weight = self.weights.get(name, 1.0 / len(self.models))
            model_probs = model.predict_proba(X)
            
            # Handle shape mismatch
            if model_probs.shape[1] != self.n_classes:
                continue
            
            probs += weight * model_probs
        
        # Renormalize
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs
    
    def predict_proba(self, X):
        if not self.is_fitted:
            return np.ones((len(X), self.n_classes)) / self.n_classes
        
        X_scaled = self.scaler.transform(X)
        
        # Use stacking meta-model if available
        if self.meta_model is not None:
            try:
                meta_features = []
                for name in sorted(self.models.keys()):
                    model = self.models[name]
                    if isinstance(model, DummyModel):
                        continue
                    preds = model.predict_proba(X_scaled)
                    if preds.shape[1] == self.n_classes:
                        meta_features.append(preds)
                
                if len(meta_features) >= 2:
                    meta_X = np.hstack(meta_features)
                    probs = self.meta_model.predict_proba(meta_X)
                else:
                    # Fallback to weighted ensemble
                    probs = self._predict_ensemble(X_scaled)
            except Exception:
                # Fallback to weighted ensemble
                probs = self._predict_ensemble(X_scaled)
        else:
            probs = self._predict_ensemble(X_scaled)
        
        # Temperature scaling
        probs = self.temp_scaler.transform(probs)
        
        return probs
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


def add_cross_asset_features(data: dict, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add BTC/ETH correlation as features"""
    df = df.copy()
    
    # Get BTC and ETH returns
    btc_df = data.get('BTC-USD')
    eth_df = data.get('ETH-USD')
    
    if btc_df is not None and symbol != 'BTC-USD':
        btc_ret = btc_df['Close'].pct_change(15).reindex(df.index)
        df['btc_ret_15m'] = btc_ret.fillna(0)
        # Rolling correlation
        df['btc_corr_1h'] = df['ret_15m'].rolling(4).corr(btc_ret).fillna(0)
    
    if eth_df is not None and symbol not in ['ETH-USD', 'BTC-USD']:
        eth_ret = eth_df['Close'].pct_change(15).reindex(df.index)
        df['eth_ret_15m'] = eth_ret.fillna(0)
    
    return df


def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-of-day cyclical features"""
    df = df.copy()
    
    if hasattr(df.index, 'hour'):
        hour = df.index.hour
    else:
        hour = pd.to_datetime(df.index).hour
    
    # Cyclical encoding (better than one-hot for time)
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # Day of week
    if hasattr(df.index, 'dayofweek'):
        dow = df.index.dayofweek
    else:
        dow = pd.to_datetime(df.index).dayofweek
    
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    
    return df


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def load_data():
    """Load 30-day market data"""
    data_path = Path(__file__).parent / 'data' / 'market_data_30d.pkl'
    if data_path.exists():
        print(f"Loading 30-day data from {data_path}")
        raw = pd.read_pickle(data_path)
        # Handle nested 'data' key format
        if isinstance(raw, dict) and 'data' in raw:
            return raw['data']
        return raw
    
    # Fallback to 7-day
    data_path = Path(__file__).parent / 'data' / 'market_data_7d.pkl'
    if data_path.exists():
        print(f"Loading 7-day data from {data_path}")
        raw = pd.read_pickle(data_path)
        if isinstance(raw, dict) and 'data' in raw:
            return raw['data']
        return raw
    
    raise FileNotFoundError("No market data found")


def train_binary_model(data, symbol, tune=True):
    """Train binary up/down model"""
    df = data[symbol]
    df = calculate_all_features(df)
    df = add_seasonality_features(df)
    
    X, y = create_training_data(df, horizon_minutes=60, binary=True)
    
    if len(X) < 100:
        print(f"  ⚠️ Insufficient data ({len(X)} rows)")
        return DummyModel(2)
    
    print(f"  Training on {len(X)} samples, features: {X.shape[1]}")
    print(f"  Class balance: {y.mean():.3f} up, {1-y.mean():.3f} down")
    
    model = EnhancedEnsembleClassifier(n_classes=2)
    model.fit(X, y, tune_hyperparams=tune, n_trials=25)
    
    # Test predictions
    probs = model.predict_proba(X[-10:])
    print(f"  Prob range: [{probs.min():.3f}, {probs.max():.3f}], mean: {probs.mean():.3f}")
    
    return model


def train_breakout_model(data, symbol, tune=True):
    """Train breakout continuation/reversal model"""
    if symbol not in data:
        print(f"  ⚠️ No data for {symbol}")
        return DummyModel(2)
    
    df = data[symbol]
    df = calculate_all_features(df)
    df = add_seasonality_features(df)
    
    X, y = create_training_data(df, horizon_minutes=60, binary=True)
    
    if len(X) < 100:
        return DummyModel(2)
    
    print(f"  Samples: {len(X)}, features: {X.shape[1]}")
    print(f"  Class balance: {y.mean():.3f} up, {1-y.mean():.3f} down")
    
    model = EnhancedEnsembleClassifier(n_classes=2)
    model.fit(X, y, tune_hyperparams=tune, n_trials=15)
    
    probs = model.predict_proba(X[-10:])
    print(f"  Prob range: [{probs.min():.3f}, {probs.max():.3f}], mean: {probs.mean():.3f}")
    
    return model


def train_hitfirst_model(data, tune=True):
    """Train HITFIRST 3-class model"""
    df = data['ETH-USD']
    df = calculate_all_features(df)
    df = add_seasonality_features(df)
    
    # Get features
    feature_cols = get_feature_columns(include_external=False)
    X = df[feature_cols].values
    
    # Create 3-class labels: 0=up_first, 1=down_first, 2=neither
    horizon = 60
    future_high = df['High'].rolling(horizon).max().shift(-horizon)
    future_low = df['Low'].rolling(horizon).min().shift(-horizon)
    current_close = df['Close']
    
    up_first = (future_high / current_close - 1) > 0.005  # 0.5% move up
    down_first = (1 - future_low / current_close) > 0.005  # 0.5% move down
    
    y = np.where(up_first & ~down_first, 0, 
                 np.where(down_first & ~up_first, 1, 2))
    
    # Remove NaN
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]
    
    if len(X) < 100:
        return DummyModel(3)
    
    print(f"  Samples: {len(X)}, features: {X.shape[1]}")
    classes, counts = np.unique(y, return_counts=True)
    print(f"  Class distribution: {dict(zip(classes, counts))}")
    
    model = EnhancedEnsembleClassifier(n_classes=3)
    model.fit(X, y, tune_hyperparams=tune, n_trials=20)
    
    return model


def train_lbfgs_model(data, symbol, tune=True):
    """Train LBFGS 5-bucket model"""
    df = data[symbol]
    df = calculate_all_features(df)
    df = add_seasonality_features(df)
    
    y_bucket, y_Q = create_lbfgs_labels(df, horizon_minutes=360)
    X = df[FEATURE_COLS].values
    
    # Filter NaN rows
    valid = ~np.isnan(y_bucket)
    X = X[valid]
    y_bucket = y_bucket[valid].astype(int)
    y_Q = y_Q[valid]
    
    if len(X) < 100:
        return {'bucket': DummyModel(5), 'Q': {}}
    
    print(f"  Samples: {len(X)}, features: {X.shape[1]}")
    
    # Bucket classifier
    bucket_model = EnhancedEnsembleClassifier(n_classes=5)
    bucket_model.fit(X, y_bucket, tune_hyperparams=tune, n_trials=15)
    
    # Q models for each bucket/threshold combo
    q_models = {}
    thresholds = [0.5, 1.0, 2.0]
    for b_idx, bucket in enumerate([0, 1, 3, 4]):
        for t_idx in range(3):
            col = b_idx * 3 + t_idx
            q_valid = ~np.isnan(y_Q[:, col])
            if q_valid.sum() < 50:
                continue
            X_q = X[q_valid]
            y_q = y_Q[q_valid, col].astype(int)
            if len(np.unique(y_q)) < 2:
                continue
            try:
                q_model = EnhancedEnsembleClassifier(n_classes=2)
                q_model.fit(X_q, y_q, tune_hyperparams=False)
                q_models[(bucket, t_idx)] = q_model
            except:
                pass
    
    print(f"  Q models trained: {len(q_models)}/12")
    return {'bucket_model': bucket_model, 'q_models': q_models}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("SN123 MANTIS - Enhanced Ensemble Training v3")
    print("=" * 60)
    
    data = load_data()
    print(f"Loaded {len(data)} symbols")
    
    # Binary models
    print("\n" + "=" * 60)
    print("BINARY MODELS")
    print("=" * 60)
    
    binary_symbols = ['ETH-USD', 'BTC-USD', 'CAD=X', 'NZD=X', 'CHF=X', 'SI=F']
    binary_models = {}
    
    for symbol in binary_symbols:
        if symbol in data:
            print(f"\n{symbol}:")
            binary_models[symbol] = train_binary_model(data, symbol, tune=True)
    
    with open(MODEL_DIR / 'simple_models_v3.pkl', 'wb') as f:
        pickle.dump(binary_models, f)
    print(f"\n💾 Saved {len(binary_models)} binary models")
    
    # Forex models
    print("\n" + "=" * 60)
    print("FOREX MODELS")
    print("=" * 60)
    
    forex_symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']
    forex_models = {}
    
    for symbol in forex_symbols:
        if symbol in data:
            print(f"\n{symbol}:")
            forex_models[symbol] = train_binary_model(data, symbol, tune=True)
    
    with open(MODEL_DIR / 'forex_models_v3.pkl', 'wb') as f:
        pickle.dump(forex_models, f)
    print(f"\n💾 Saved {len(forex_models)} forex models")
    
    # HITFIRST
    print("\n" + "=" * 60)
    print("HITFIRST MODEL")
    print("=" * 60)
    
    hitfirst_model = train_hitfirst_model(data, tune=True)
    with open(MODEL_DIR / 'hitfirst_model_v3.pkl', 'wb') as f:
        pickle.dump(hitfirst_model, f)
    print("💾 Saved HITFIRST model")
    
    # LBFGS
    print("\n" + "=" * 60)
    print("LBFGS MODELS")
    print("=" * 60)
    
    lbfgs_models = {}
    for symbol in ['BTC-USD', 'ETH-USD']:
        if symbol in data:
            print(f"\n{symbol}:")
            lbfgs_models[symbol] = train_lbfgs_model(data, symbol, tune=True)
    
    with open(MODEL_DIR / 'lbfgs_models_v3.pkl', 'wb') as f:
        pickle.dump(lbfgs_models, f)
    print(f"\n💾 Saved {len(lbfgs_models)} LBFGS models")
    
    # MULTI-BREAKOUT
    print("\n" + "=" * 60)
    print("MULTI-BREAKOUT MODELS")
    print("=" * 60)
    
    breakout_symbols = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD',
        'ADA-USD', 'AVAX-USD', 'DOGE-USD', 'DOT-USD', 'LINK-USD',
        'MATIC-USD', 'SHIB-USD', 'LTC-USD', 'UNI-USD', 'ATOM-USD',
        'XLM-USD', 'ETC-USD', 'FIL-USD', 'HBAR-USD', 'ICP-USD',
        'VET-USD', 'NEAR-USD', 'ALGO-USD', 'FTM-USD', 'AAVE-USD',
        'EOS-USD', 'SAND-USD', 'MANA-USD', 'AXS-USD', 'THETA-USD',
        'XTZ-USD', 'TRX-USD', 'APT-USD'
    ]
    
    breakout_models = {}
    for i, symbol in enumerate(breakout_symbols):
        if symbol in data:
            print(f"\n[{i+1}/{len(breakout_symbols)}] {symbol}:")
            breakout_models[symbol] = train_breakout_model(data, symbol, tune=True)
        else:
            print(f"\n[{i+1}/{len(breakout_symbols)}] {symbol}: ⚠️ No data")
    
    with open(MODEL_DIR / 'multibreakout_models_v3.pkl', 'wb') as f:
        pickle.dump(breakout_models, f)
    print(f"\n💾 Saved {len(breakout_models)} breakout models")
    
    print("\n" + "=" * 60)
    print("✅ V3 TRAINING COMPLETE!")
    print("=" * 60)
