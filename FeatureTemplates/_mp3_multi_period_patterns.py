import pandas as pd
import numpy as np

METADATA = {
    "name": "mp3_multi_period_patterns",
    "description": (
        "Per-ticker proxy for MP3 (Multi-Period Pattern Pre-training). "
        "Uses sliding-window FFT on log-returns to extract spectral power and phase "
        "at weekly (5d), monthly (21d), and quarterly (63d) cycles, then computes "
        "cross-period phase alignment features that operationalise the paper's "
        "'temporal mirage' insight: similar short windows diverge in future trend "
        "when their multi-period phase context differs. Cross-stock spatial modelling "
        "and learned pre-training are omitted â€” they require simultaneous access to "
        "multiple tickers and a training phase, neither of which is available here."
    ),
    "requires": ["Close"],
    "produces": [
        "mp3_power_5", "mp3_power_21", "mp3_power_63",
        "mp3_phase_5", "mp3_phase_21", "mp3_phase_63",
        "mp3_align_5_21", "mp3_align_21_63", "mp3_align_5_63",
        "mp3_mirage_score", "mp3_dominant_period",
    ],
    "tags": ["multi-period", "spectral", "fft", "pattern", "temporal-mirage"],
    "version": "1.0",
    "author": "MP3 proxy â€” sliding-window FFT decomposition; spatial/pre-training omitted",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].values.astype(np.float64)
    n = len(close)

    # Log returns; index 0 is always NaN
    ret = np.empty(n)
    ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        ret[1:] = np.log(close[1:] / close[:-1])

    # Rolling FFT window: 2 quarters captures all three target periods cleanly
    WIN = 126
    PERIODS = [5, 21, 63]
    n_bins = WIN // 2 + 1
    # Discrete FFT bin closest to each period (freq = WIN/period)
    freq_idx = {p: min(max(round(WIN / p), 1), n_bins - 1) for p in PERIODS}

    power_arr = {p: np.full(n, np.nan) for p in PERIODS}
    phase_arr = {p: np.full(n, np.nan) for p in PERIODS}

    # First valid window uses ret[1:WIN+1]; result lands at row index WIN
    if n > WIN:
        ret_valid = ret[1:]   # length n-1, skips the NaN at index 0
        # wins[i] = ret[i+1 : i+1+WIN], placed at output row i+WIN
        wins = np.lib.stride_tricks.sliding_window_view(ret_valid, WIN)
        ffts = np.fft.rfft(wins, axis=1)   # shape (n-WIN, n_bins), complex
        for p in PERIODS:
            fi = freq_idx[p]
            coeff = ffts[:, fi]
            power_arr[p][WIN:] = np.abs(coeff) / WIN
            phase_arr[p][WIN:] = np.angle(coeff)

    start = WIN

    align_5_21  = np.full(n, np.nan)
    align_21_63 = np.full(n, np.nan)
    align_5_63  = np.full(n, np.nan)
    mirage      = np.full(n, np.nan)
    dominant    = np.full(n, np.nan)

    if n > start:
        ph5  = phase_arr[5][start:]
        ph21 = phase_arr[21][start:]
        ph63 = phase_arr[63][start:]
        pw5  = power_arr[5][start:]
        pw21 = power_arr[21][start:]
        pw63 = power_arr[63][start:]

        # Cross-period phase coherence: cos(Î”phase) âˆˆ [-1, 1]
        # +1 = perfectly in-phase, -1 = perfectly anti-phase
        align_5_21[start:]  = np.cos(ph5  - ph21)
        align_21_63[start:] = np.cos(ph21 - ph63)
        align_5_63[start:]  = np.cos(ph5  - ph63)

        # Temporal mirage score: misalignment of short (5d) vs long (63d) cycles,
        # weighted by 63d spectral power (high when a real quarterly cycle exists
        # but contradicts the short-term pattern â€” the "mirage" condition)
        mirage[start:] = (1.0 - align_5_63[start:]) * 0.5 * pw63

        # Which period currently dominates the return spectrum?
        stacked = np.stack([pw5, pw21, pw63], axis=1)
        dominant[start:] = np.array([5.0, 21.0, 63.0])[np.argmax(stacked, axis=1)]

    df["mp3_power_5"]         = power_arr[5]
    df["mp3_power_21"]        = power_arr[21]
    df["mp3_power_63"]        = power_arr[63]
    df["mp3_phase_5"]         = phase_arr[5]
    df["mp3_phase_21"]        = phase_arr[21]
    df["mp3_phase_63"]        = phase_arr[63]
    df["mp3_align_5_21"]      = align_5_21
    df["mp3_align_21_63"]     = align_21_63
    df["mp3_align_5_63"]      = align_5_63
    df["mp3_mirage_score"]    = mirage
    df["mp3_dominant_period"] = dominant

    return df