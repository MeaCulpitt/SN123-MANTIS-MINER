**Processing `auto_retrain.py` and `.env` template**

I'm checking `auto_retrain.py`. It's a nicely self-contained orchestrator that calls `data_collector_v3.py`, `train_ensemble_v3.py`, and `ml_miner_v8.py`. It has no secrets inside.

For the `.env` file, this is critical because it tells them what environmental variables the scripts expect, *but* it has real credentials (R2 keys and an account ID!). I need to produce an `.env.example` file that reds-out the real secrets.

The real `.env` is:
```env
R2_ACCESS_KEY_ID=f8046556985808b287fbb0167b4ef255
R2_SECRET_ACCESS_KEY=931cef628a4881de35273a5169717e7c229a5a50c34115475e0c3890b6a86769
R2_ENDPOINT=https://5dbd6b5038744b87e5e8605e6a0c4dc8.r2.cloudflarestorage.com
R2_BUCKET_NAME=mantis-miner
R2_PUBLIC_URL=https://pub-e64ed1f6f7354401a68e54a859e764a9.r2.dev/5FTPaM4eWttbwJ3eTL1R1k5Ty86FYz96AUkgg3K4nCoBuqni
WALLET_NAME=tao_miner
WALLET_HOTKEY=miner_hotkey
```

When I present this, I will mask all the values with `[REDACTED_...]`.

I will output `auto_retrain.py` and `.env.example`. This should wrap up the "how to run the pipeline" set of scripts!**Finalizing output**

I will format the markdown output with both `auto_retrain.py` (the run orchestrator) and `.env.example` (the required settings to make the miner and R2 bucket connection work). I'll keep the response snappy and conversational.

Let's send this batch!
Here are the final configuration pieces to automate the training and execution pipeline: the **`.env` template** and **`auto_retrain.py`**.

### 1. `.env` (Environment Variables)
*This file powers the `comms.py` and `ml_miner_v8.py` R2 storage capability. I have safely sanitized your real access keys and wallet details.*

```env
# R2 Bucket Credentials for Validator Payloads
R2_ACCESS_KEY_ID=[REDACTED_ACCESS_KEY]
R2_SECRET_ACCESS_KEY=[REDACTED_SECRET_KEY]
R2_ENDPOINT=[REDACTED_CLOUDFLARE_ENDPOINT]
R2_BUCKET_NAME=mantis-miner
R2_PUBLIC_URL=[REDACTED_PUBLIC_R2_DEV_URL]/[REDACTED_HOTKEY_ADDRESS]

# Bittensor Wallet
WALLET_NAME=tao_miner
WALLET_HOTKEY=miner_hotkey
```

---

### 2. `auto_retrain.py`
*This is the system daemon that runs the `data_collector`, triggers `train_ensemble_v3.py`, and dynamically restarts `ml_miner_v8.py` so the miner constantly runs on fresh model weights.*

```python
#!/usr/bin/env python3
"""
SN123 MANTIS - Auto Retraining Script
Collects fresh data and retrains models every N hours.
Run as a daemon or cron job.
"""

import os
import sys
import time
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('auto_retrain.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
VENV_PYTHON = BASE_DIR / '.venv' / 'bin' / 'python'

# Retraining interval in hours
RETRAIN_INTERVAL_HOURS = 6

# Data collection scripts
DATA_COLLECTOR = BASE_DIR / 'data_collector_v3.py'
TRAINER = BASE_DIR / 'train_ensemble_v3.py'
MINER = BASE_DIR / 'ml_miner_v8.py'


def run_script(script_path, timeout=1800):
    """Run a Python script and return success status"""
    log.info(f"Running {script_path.name}...")
    
    try:
        result = subprocess.run(
            [str(VENV_PYTHON), str(script_path)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            log.info(f"✅ {script_path.name} completed successfully")
            return True
        else:
            log.error(f"❌ {script_path.name} failed: {result.stderr[-500:]}")
            return False
            
    except subprocess.TimeoutExpired:
        log.error(f"⏰ {script_path.name} timed out after {timeout}s")
        return False
    except Exception as e:
        log.error(f"❌ {script_path.name} error: {e}")
        return False


def collect_fresh_data():
    """Collect fresh market data"""
    return run_script(DATA_COLLECTOR, timeout=600)


def retrain_models():
    """Retrain all ML models"""
    return run_script(TRAINER, timeout=1800)


def restart_miner():
    """Restart the miner with new models"""
    log.info("Restarting miner...")
    
    # Kill existing miner
    subprocess.run(['pkill', '-f', 'ml_miner_v[78].py'], capture_output=True)
    time.sleep(2)
    
    # Start new miner
    subprocess.Popen(
        [str(VENV_PYTHON), '-u', str(MINER)],
        cwd=str(BASE_DIR),
        stdout=open(BASE_DIR / 'miner_v8.log', 'a'),
        stderr=subprocess.STDOUT,
        start_new_session=True
    )
    
    log.info("✅ Miner restarted")


def retrain_cycle():
    """Run one full retrain cycle"""
    log.info("=" * 60)
    log.info(f"Starting retrain cycle at {datetime.utcnow().isoformat()}")
    log.info("=" * 60)
    
    # Step 1: Collect fresh data
    if not collect_fresh_data():
        log.warning("Data collection failed, using existing data")
    
    # Step 2: Retrain models
    if retrain_models():
        # Step 3: Restart miner only if training succeeded
        restart_miner()
    else:
        log.error("Training failed, keeping existing models")
    
    log.info(f"Retrain cycle complete at {datetime.utcnow().isoformat()}")


def daemon_mode():
    """Run as daemon, retraining every N hours"""
    log.info(f"Starting auto-retrain daemon (interval: {RETRAIN_INTERVAL_HOURS}h)")
    
    while True:
        try:
            retrain_cycle()
        except Exception as e:
            log.error(f"Retrain cycle error: {e}")
        
        # Sleep until next cycle
        sleep_seconds = RETRAIN_INTERVAL_HOURS * 3600
        log.info(f"Sleeping {RETRAIN_INTERVAL_HOURS}h until next retrain...")
        time.sleep(sleep_seconds)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SN123 Auto Retrainer')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    parser.add_argument('--once', action='store_true', help='Run one cycle and exit')
    parser.add_argument('--interval', type=int, default=6, help='Hours between retrains')
    args = parser.parse_args()
    
    global RETRAIN_INTERVAL_HOURS
    RETRAIN_INTERVAL_HOURS = args.interval
    
    if args.daemon:
        daemon_mode()
    elif args.once:
        retrain_cycle()
    else:
        print("Usage: auto_retrain.py --daemon | --once")
        print("  --daemon: Run continuously, retrain every N hours")
        print("  --once: Run one retrain cycle and exit")
        print("  --interval N: Hours between retrains (default: 6)")


if __name__ == '__main__':
    main()
```

If you ever need to clone this node to a new server, those are the exact pieces that glue data collection, training, and routing together! Are there any specific mathematical feature components (like `data_collector_v3.py`) that you also want, or does this complete your export?
