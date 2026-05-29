"""
Production data pipeline: BTC/USD + XAU/USD (5m) → compound quadrant tokens (0–15).

Ingests historical OHLCV CSVs, time-aligns assets, engineers volatility/momentum
quadrants, merges into a joint token stream, and exports chronologically split
train/validation arrays.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

VOCAB_SIZE: Final[int] = 16
TRAIN_RATIO: Final[float] = 0.90
DOMINANCE_WARN_THRESHOLD: Final[float] = 0.40
MIN_ACTIVE_TOKENS: Final[int] = 12


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for the tokenization pipeline."""

    btc_path: Path = Path("data/raw/BTCUSD_1m_Combined_Index.csv")
    xau_path: Path = Path("data/raw/XAU_5m_data.csv")
    output_dir: Path = Path("data/processed")
    rolling_window: int = 24
    bar_freq: str = "5min"
    train_ratio: float = TRAIN_RATIO


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and enforce a UTC-naive DatetimeIndex."""
    col_map = {c.lower().strip(): c for c in df.columns}
    rename: dict[str, str] = {}
    for canonical, aliases in (
        ("open", ("open",)),
        ("high", ("high",)),
        ("low", ("low",)),
        ("close", ("close",)),
        ("volume", ("volume", "vol")),
    ):
        for alias in aliases:
            if alias in col_map:
                rename[col_map[alias]] = canonical
                break

    time_col = None
    for candidate in ("timestamp", "date", "open time", "datetime", "time"):
        if candidate in col_map:
            time_col = col_map[candidate]
            break
    if time_col is None:
        raise ValueError(f"No timestamp column found in columns: {list(df.columns)}")

    out = df.rename(columns=rename).copy()
    out["timestamp"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"])
    out["timestamp"] = out["timestamp"].dt.tz_convert(None)
    out = out.set_index("timestamp").sort_index()
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns after normalization: {missing}")
    return out[required].astype(np.float64)


def load_btc_csv(path: Path) -> pd.DataFrame:
    """Load BTC/USD CSV (1m or 5m) and normalize to 5-minute bars."""
    logger.info("Loading BTC from %s", path)
    df = pd.read_csv(path, low_memory=False)
    ohlcv = _normalize_ohlcv(df)

    inferred = pd.infer_freq(ohlcv.index[: min(len(ohlcv), 5000)])
    if inferred is None:
        median_delta = ohlcv.index.to_series().diff().median()
        if median_delta <= pd.Timedelta("2min"):
            inferred = "1min"
        else:
            inferred = "5min"

    if inferred == "1min" or "1m" in path.name.lower():
        logger.info("Resampling BTC 1m → 5m (%d rows before)", len(ohlcv))
        ohlcv = (
            ohlcv.resample("5min")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna(subset=["close"])
        )
    return ohlcv


def load_xau_csv(path: Path) -> pd.DataFrame:
    """Load XAU/USD 5-minute CSV (semicolon-separated vendor format)."""
    logger.info("Loading XAU from %s", path)
    df = pd.read_csv(
        path,
        sep=";",
        parse_dates=["Date"],
        date_format="%Y.%m.%d %H:%M",
        low_memory=False,
    )
    df = df.rename(columns={"Date": "timestamp"})
    return _normalize_ohlcv(df)


# ---------------------------------------------------------------------------
# Time synchronization
# ---------------------------------------------------------------------------


def synchronize_assets(
    btc: pd.DataFrame,
    xau: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join BTC and XAU on timestamp (drops gold-closed sessions / weekends)
    and forward-fills minor gaps in OHLCV columns.
    """
    btc_renamed = btc.add_prefix("btc_")
    xau_renamed = xau.add_prefix("xau_")

    aligned = btc_renamed.join(xau_renamed, how="inner")
    price_cols = list(aligned.columns)
    aligned[price_cols] = aligned[price_cols].ffill()
    aligned = aligned.dropna(subset=["btc_close", "xau_close"])
    logger.info(
        "Synchronized %d overlapping 5m bars (BTC range: %s → %s)",
        len(aligned),
        aligned.index.min(),
        aligned.index.max(),
    )
    return aligned


# ---------------------------------------------------------------------------
# Technical quadrant engineering
# ---------------------------------------------------------------------------


def _rolling_vwap(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    pv = (close * volume).rolling(window, min_periods=window)
    vol_sum = volume.rolling(window, min_periods=window).sum()
    return pv.sum() / vol_sum.replace(0, np.nan)


def engineer_features(
    close: pd.Series,
    volume: pd.Series,
    window: int,
) -> pd.DataFrame:
    """Log returns, rolling volatility, and momentum (distance from rolling VWAP)."""
    log_returns = np.log(close / close.shift(1))
    volatility = log_returns.rolling(window, min_periods=window).std()
    vwap = _rolling_vwap(close, volume, window)
    momentum = close - vwap
    sma = close.rolling(window, min_periods=window).mean()
    return pd.DataFrame(
        {
            "log_return": log_returns,
            "volatility": volatility,
            "momentum": momentum,
            "sma": sma,
        },
        index=close.index,
    )


def lag_features_for_decision(
    close: pd.Series,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Shift rolling features (and close) by one bar so quadrant t uses only
    information available after the prior bar has closed.
    """
    return pd.DataFrame(
        {
            "log_return": features["log_return"].shift(1),
            "volatility": features["volatility"].shift(1),
            "momentum": features["momentum"].shift(1),
            "sma": features["sma"].shift(1),
            "close": close.shift(1),
        },
        index=features.index,
    )


def fit_train_vol_medians(
    btc_vol: pd.Series,
    xau_vol: pd.Series,
    train_end_idx: int,
) -> tuple[float, float]:
    """
    Estimate static volatility medians from the chronological training slice only
    (vol_med_btc, vol_med_gold) to avoid validation lookahead.
    """
    vol_med_btc = float(btc_vol.iloc[:train_end_idx].dropna().median())
    vol_med_gold = float(xau_vol.iloc[:train_end_idx].dropna().median())
    return vol_med_btc, vol_med_gold


def assign_quadrant(
    lagged: pd.DataFrame,
    vol_median: float,
) -> pd.Series:
    """
    Discretize into 4 quadrants:
      0 = Low Vol / Bearish
      1 = Low Vol / Bullish
      2 = High Vol / Bearish
      3 = High Vol / Bullish

    Vol regime: static training median (no validation leakage).
    Direction: lagged close vs lagged rolling SMA.
    """
    high_vol = lagged["volatility"] >= vol_median
    bullish = lagged["close"] >= lagged["sma"]
    quadrant = high_vol.astype(np.int8) * 2 + bullish.astype(np.int8)
    return quadrant


def build_compound_tokens(
    aligned: pd.DataFrame,
    window: int,
    train_ratio: float = TRAIN_RATIO,
) -> np.ndarray:
    """Compute per-asset quadrants and merge: token = btc_state * 4 + xau_state."""
    train_end_idx = int(len(aligned) * train_ratio)

    btc_features = engineer_features(
        aligned["btc_close"], aligned["btc_volume"], window
    )
    xau_features = engineer_features(
        aligned["xau_close"], aligned["xau_volume"], window
    )

    btc_lagged = lag_features_for_decision(aligned["btc_close"], btc_features)
    xau_lagged = lag_features_for_decision(aligned["xau_close"], xau_features)

    vol_med_btc, vol_med_gold = fit_train_vol_medians(
        btc_lagged["volatility"],
        xau_lagged["volatility"],
        train_end_idx,
    )
    logger.info(
        "Training-only vol medians — vol_med_btc=%.8f, vol_med_gold=%.8f (n=%d bars)",
        vol_med_btc,
        vol_med_gold,
        train_end_idx,
    )

    btc_state = assign_quadrant(btc_lagged, vol_med_btc)
    xau_state = assign_quadrant(xau_lagged, vol_med_gold)

    valid = btc_state.notna() & xau_state.notna()
    tokens = (btc_state[valid].astype(np.int16) * 4 + xau_state[valid].astype(np.int16)).values
    logger.info("Built %d compound tokens after rolling warmup", len(tokens))
    return tokens


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def chronological_split(
    tokens: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """Split tokens in time order (no shuffle)."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    split_idx = int(len(tokens) * train_ratio)
    if split_idx == 0 or split_idx == len(tokens):
        raise ValueError(
            f"Split index {split_idx} invalid for {len(tokens)} tokens; "
            "dataset too small for requested ratio."
        )
    return tokens[:split_idx], tokens[split_idx:]


def export_tokens(
    train: np.ndarray,
    val: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Persist train/validation token arrays as .npy files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_tokens.npy"
    val_path = output_dir / "val_tokens.npy"
    np.save(train_path, train)
    np.save(val_path, val)
    logger.info("Saved %s (%d tokens)", train_path, len(train))
    logger.info("Saved %s (%d tokens)", val_path, len(val))
    return train_path, val_path


def run_pipeline(config: PipelineConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Execute the full ingestion → tokenization → export pipeline."""
    cfg = config or PipelineConfig()
    btc = load_btc_csv(cfg.btc_path)
    xau = load_xau_csv(cfg.xau_path)
    aligned = synchronize_assets(btc, xau)
    tokens = build_compound_tokens(aligned, cfg.rolling_window, cfg.train_ratio)
    train, val = chronological_split(tokens, cfg.train_ratio)
    export_tokens(train, val, cfg.output_dir)
    return train, val


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------


def _token_distribution(tokens: np.ndarray) -> dict[int, int]:
    unique, counts = np.unique(tokens, return_counts=True)
    return {int(t): int(c) for t, c in zip(unique, counts)}


def verify_and_report(
    train: np.ndarray,
    val: np.ndarray,
    dominance_threshold: float = DOMINANCE_WARN_THRESHOLD,
    min_active_tokens: int = MIN_ACTIVE_TOKENS,
) -> None:
    """Assert token integrity and print distribution diagnostics."""
    for name, arr in (("train_tokens", train), ("val_tokens", val)):
        assert not np.isnan(arr).any(), f"{name} contains NaN values"
        assert not np.isinf(arr).any(), f"{name} contains infinite values"

    combined = np.concatenate([train, val])
    assert combined.min() >= 0, f"Token min out of range: {combined.min()}"
    assert combined.max() <= 15, f"Token max out of range: {combined.max()}"

    dist = _token_distribution(combined)
    total = len(combined)

    print("\n=== Compound Token Distribution ===")
    print(f"{'Token':>6} | {'Count':>10} | {'Pct':>8}")
    print("-" * 30)
    for token in sorted(dist):
        count = dist[token]
        pct = 100.0 * count / total
        print(f"{token:>6} | {count:>10} | {pct:7.2f}%")
    print(f"\nUnique active tokens: {len(dist)} / {VOCAB_SIZE}")

    for token, count in dist.items():
        share = count / total
        if share > dominance_threshold:
            warnings.warn(
                f"Token {token} dominates {share:.1%} of the dataset "
                f"(>{dominance_threshold:.0%} threshold). "
                "Consider rebalancing features or expanding data.",
                UserWarning,
                stacklevel=2,
            )

    if len(dist) < min_active_tokens:
        warnings.warn(
            f"Only {len(dist)} unique tokens active (expected >= {min_active_tokens}). "
            "Quadrant boundaries may be too coarse or assets overly correlated.",
            UserWarning,
            stacklevel=2,
        )

    print("\nVerification passed: finite tokens in [0, 15].")


if __name__ == "__main__":
    train_tokens, val_tokens = run_pipeline()
    verify_and_report(train_tokens, val_tokens)
