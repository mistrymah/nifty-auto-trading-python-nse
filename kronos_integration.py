import sys
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from pathlib import Path
import pandas as pd
import torch

kronos_model_path = Path(__file__).parent.parent / "Kronos" / "model"
sys.path.insert(0, str(kronos_model_path))
sys.path.insert(0, str(Path(__file__).parent.parent / "Kronos"))

from kronos import KronosPredictor


class NiftyKronosPredictor:
    def __init__(self, model_name: str = "NeoQuasar/Kronos-small"):
        from kronos import KronosTokenizer, Kronos

        print("Loading tokenizer...")
        self.tokenizer = KronosTokenizer.from_pretrained(
            "NeoQuasar/Kronos-Tokenizer-base"
        )
        print("Loading model...")
        self.model = Kronos.from_pretrained(model_name)
        self.predictor = KronosPredictor(self.model, self.tokenizer, max_context=512)
        print("Kronos loaded successfully")

    def forecast(self, df: pd.DataFrame, lookback: int = 300, pred_len: int = 30):
        """Forecast future prices using the latest available candles.
        
        Takes the most recent `lookback` candles as context and generates
        `pred_len` future timestamps based on the detected candle frequency.
        """
        lookback = int(lookback)
        pred_len = int(pred_len)
        
        if df.empty:
            raise ValueError("Input dataframe is empty.")
        
        # Use the latest available candles (not from the start of history)
        actual_lookback = min(lookback, len(df))
        x_df = df.tail(actual_lookback).reset_index(drop=True)
        x_timestamp = x_df["timestamps"].reset_index(drop=True)

        # Detect candle frequency from the last two timestamps
        if actual_lookback >= 2:
            freq = x_df["timestamps"].iloc[-1] - x_df["timestamps"].iloc[-2]
        else:
            freq = pd.Timedelta(minutes=5)  # safe fallback

        # Generate actual future timestamps (from now, not historical data)
        last_timestamp = x_df["timestamps"].iloc[-1]
        y_timestamps = [last_timestamp + freq * (i + 1) for i in range(pred_len)]
        y_timestamp = pd.Series(y_timestamps)

        # Extract only the price/volume columns for the predictor
        feature_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in x_df.columns]
        x_features = x_df[feature_cols].reset_index(drop=True)

        print(f"Input: {len(x_features)} candles, predicting {pred_len} ahead")

        pred_df = self.predictor.predict(
            df=x_features,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )
        return pred_df


if __name__ == "__main__":
    from fetcher import UpstoxFetcher

    fetcher = UpstoxFetcher()
    df = fetcher.get_nifty_historical("5m", "5d")

    print(f"Loaded {len(df)} historical candles")
    print("Loading Kronos model...")

    predictor = NiftyKronosPredictor()
    print("Making prediction...")

    pred = predictor.forecast(df, lookback=300, pred_len=30)
    print("Prediction:")
    print(pred.head())
