import librosa
import numpy as np
import pandas as pd
import math
from scipy.ndimage import uniform_filter1d


def merge_events(df, flag_col, min_duration=0.0):
    rows = df if isinstance(df, list) else df.to_dict(orient="records")
    merged = []
    current = None

    for row in rows:
        if row.get(flag_col) == 1:
            if current is None:
                current = {"start": row["start"], "end": row["end"]}
            else:
                current["end"] = row["end"]
        else:
            if current:
                if current["end"] - current["start"] >= min_duration:
                    merged.append(current)
                current = None

    if current and current["end"] - current["start"] >= min_duration:
        merged.append(current)

    return merged


def analyze_temporal_acoustics(audio_path, calibration_sec=5.0, window_sec=1.0):
    HOP_LENGTH = 512
    SILENCE_TH = 0.25
    LOW_TH = 0.7
    HIGH_TH = 1.3
    SMOOTHING_SIZE = 5

    y, sr = librosa.load(audio_path, sr=16000)
    total_duration = len(y) / sr

    if total_duration <= calibration_sec:
        calibration_sec = total_duration * 0.3

    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    f0, _, _ = librosa.pyin(y=y, fmin=50, fmax=500, hop_length=HOP_LENGTH)

    rms = uniform_filter1d(rms, size=SMOOTHING_SIZE)
    f0_nan_mask = np.isnan(f0)
    f0_smoothed = uniform_filter1d(np.nan_to_num(f0), size=SMOOTHING_SIZE)
    f0_smoothed[f0_nan_mask] = np.nan
    f0 = f0_smoothed

    min_len = min(len(rms), len(f0))
    rms = rms[:min_len]
    f0 = f0[:min_len]

    times = librosa.frames_to_time(np.arange(min_len), sr=sr, hop_length=HOP_LENGTH)

    calib_mask = times <= calibration_sec
    calib_rms_raw = rms[calib_mask]
    silence_floor = np.percentile(calib_rms_raw, 30)
    calib_rms_voiced = calib_rms_raw[calib_rms_raw > silence_floor]
    baseline_rms = float(np.mean(calib_rms_voiced)) if len(calib_rms_voiced) > 0 else float(np.mean(rms))
    baseline_rms = max(baseline_rms, 1e-6)

    calib_f0 = f0[calib_mask]
    calib_f0_valid = calib_f0[~np.isnan(calib_f0)]

    if len(calib_f0_valid) > 30:
        baseline_f0_std = float(np.std(calib_f0_valid))
    else:
        f0_all_valid = f0[~np.isnan(f0)]
        baseline_f0_std = float(np.std(f0_all_valid)) if len(f0_all_valid) > 0 else 1.0
    baseline_f0_std = max(baseline_f0_std, 1e-6)

    num_windows = math.ceil(total_duration / window_sec)
    data = []

    for i in range(num_windows):
        start = i * window_sec
        end = min((i + 1) * window_sec, total_duration)
        mask = (times >= start) & (times < end)

        c_rms = rms[mask]
        c_f0 = f0[mask]

        mean_rms = float(np.mean(c_rms)) if len(c_rms) > 0 else 0.0
        energy_ratio = mean_rms / baseline_rms
        c_f0_valid = c_f0[~np.isnan(c_f0)]

        if len(c_f0_valid) > 10:
            std_f0 = float(np.std(c_f0_valid))
            pitch_ratio = round(std_f0 / baseline_f0_std, 2)
        else:
            pitch_ratio = None

        is_silence = mean_rms < (baseline_rms * SILENCE_TH)

        if is_silence:
            energy_status = "silence"
        elif energy_ratio < LOW_TH:
            energy_status = "low"
        elif energy_ratio > HIGH_TH:
            energy_status = "high"
        else:
            energy_status = "normal"

        if is_silence:
            pitch_status = "no_voice"
        elif pitch_ratio is None:
            pitch_status = "uncertain"
        elif pitch_ratio < 0.6:
            pitch_status = "monotone"
        elif pitch_ratio > 1.5:
            pitch_status = "dynamic"
        else:
            pitch_status = "normal"

        # Filler tespiti:
        # - monotone: algılanabilir ama düz ses → klasik ıı/ee
        # - uncertain + sessizlik komşusu DEĞİLSE: konuşma ortasında belirsiz ses → filler olabilir
        prev_energy_status = data[-1]["energy_status"] if data else "silence"
        is_filler_monotone = (
            not is_silence and
            energy_status == "low" and
            pitch_status == "monotone"
        )
        is_filler_uncertain = (
            not is_silence and
            energy_status == "low" and
            pitch_status == "uncertain" and
            prev_energy_status not in ["silence"]
        )
        is_filler_candidate = is_filler_monotone or is_filler_uncertain

        prev_energy = data[-1]["energy_ratio"] if data else energy_ratio
        delta_energy = energy_ratio - prev_energy

        pitch_ratio_val = pitch_ratio if pitch_ratio is not None else 0.0
        is_emphasis = (
            energy_ratio > 1.5 and
            delta_energy > 0.3 and
            (pitch_status == "dynamic" or pitch_ratio_val > 1.2)
        )

        # NaN'ı tamamen engelle: pitch_ratio None ise JSON'a null yaz
        safe_pitch = round(pitch_ratio, 2) if (pitch_ratio is not None and not math.isnan(pitch_ratio)) else None

        data.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "energy_ratio": round(energy_ratio, 2),
            "delta_energy": round(delta_energy, 2),
            "pitch_ratio": safe_pitch,          # None → JSON null, asla NaN değil
            "energy_status": energy_status,
            "pitch_status": pitch_status,
            "filler_candidate": int(is_filler_candidate),
            "emphasis": int(is_emphasis)
        })

    df = pd.DataFrame(data)

    emphasis_events = merge_events(df, flag_col="emphasis", min_duration=1.0)
    filler_events = merge_events(df, flag_col="filler_candidate", min_duration=0.0)
    filler_duration = sum(e["end"] - e["start"] for e in filler_events)

    # Pandas to_dict() None → float('nan') çevirir, JSON'da geçersiz olur.
    # _clean_nan ile tüm nan'ları None'a döndür.
    def _clean_nan(records):
        for row in records:
            for k, v in row.items():
                if isinstance(v, float) and math.isnan(v):
                    row[k] = None
        return records

    temporal_data = _clean_nan(df.to_dict(orient="records"))

    return {
        "df": df,
        "temporal_data": temporal_data,
        "emphasis_events": emphasis_events,
        "filler_events": filler_events,
        "filler_duration": round(filler_duration, 2),
    }