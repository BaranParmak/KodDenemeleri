import whisper
import librosa
import re
import math
import pandas as pd

# =====================================================================
# FILLER_WORDS — Türkçe iş görüşmesi filler listesi
#
# Whisper'ın ıı sesini yazma varyasyonları (test edilmiş):
#   "Iıı" → lower() → "iıı" → normalize → "ıı"  ✓
#   "ınınından" → direkt FILLER_WORDS'te             ✓
#   "ınınının"  → direkt FILLER_WORDS'te             ✓
#
# eee sesi: Whisper çoğunlukla siliyor.
#   Acoustic fallback devreye girer (düşük güvenilirlikle).
#   Tez limitasyonu: yüksek enerjili vocalic filler'lar acoustic ile tespit edilemiyor.
# =====================================================================
FILLER_WORDS = [
    # Vocalic (yüksek) — eee
    "eeee", "eee", "ee",
    # Vocalic (alçak) — ıı: Whisper varyasyonları
    "ınınından",    # ı-n-ı-n-ı-n: en yaygın Whisper varyasyonu
    "ınınının",     # varyasyon
    "ıınınından",   # uzun varyasyon
    "ıınınından",   # uzun varyasyon 2
    "ıı ıı ıı",    # boşluklu uzun
    "ıı ıı",        # boşluklu kısa
    "ıı",           # standart
    # Kelime fillerlar
    "şey", "yani", "hani",
    # Diğer
    "mmm", "mm", "hmm", "ımm", "umm",
    "ıh", "hıh", "huh", "uh",
]

SHORT_FILLERS = {"ee", "ıı", "mm", "uh", "ıh"}


def _normalize_fillers(text):
    """
    Whisper'ın ıı sesini normalize eder.
    Sıra önemli: önce lower() çağrılmış olmalı.
    "Iıı" → lower → "iıı" → normalize → "ıı"
    [iı]{2,} pattern'i Latin I ve Türkçe ı karışımını yakalar.
    """
    text = re.sub(r'[iı]{2,}', 'ıı', text)
    text = re.sub(r'e{4,}', 'eee', text)
    return text


def _clean_transcript(text):
    """Kelime sayımı için transcript temizleme. Filler sayımından SONRA çağrılmalı."""
    text = re.sub(r'[A-ZÇŞİĞÜÖ][a-zçşığüö]+\s+[A-ZÇŞİĞÜÖ]\.\s*[A-ZÇŞİĞÜÖ]\.', '', text)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    return text.strip()


def _make_tr_boundary_pattern(filler):
    tr_alpha = r'a-zçşığüöA-ZÇŞİĞÜÖ0-9'
    if filler in SHORT_FILLERS:
        return r'(?:^|\s)' + re.escape(filler) + r'(?=\s|[,\.!?\-;:]|$)'
    else:
        return r'(?<![' + tr_alpha + r'])' + re.escape(filler) + r'(?![' + tr_alpha + r'])'


def count_filler_words(text, fillers):
    """Sıra: lower() → normalize() → say(). Returns: (total, counts)"""
    text = _normalize_fillers(text.lower())
    total = 0
    counts = {}
    for filler in fillers:
        pattern = _make_tr_boundary_pattern(filler)
        found = len(re.findall(pattern, text, re.MULTILINE))
        if found > 0:
            counts[filler] = found
        total += found
    return total, counts


def merge_filler_windows(temporal_data):
    windows = temporal_data if isinstance(temporal_data, list) else temporal_data.to_dict(orient="records")
    merged = []
    current = None
    for row in windows:
        if row.get("filler_candidate") == 1:
            if current is None:
                current = {"start": row["start"], "end": row["end"]}
            else:
                current["end"] = row["end"]
        else:
            if current:
                merged.append(current)
                current = None
    if current:
        merged.append(current)
    return merged


def estimate_fillers_from_audio(temporal_data):
    if temporal_data is None:
        return 0
    has_flag = (
        (isinstance(temporal_data, pd.DataFrame) and "filler_candidate" in temporal_data.columns) or
        (isinstance(temporal_data, list) and len(temporal_data) > 0 and "filler_candidate" in temporal_data[0])
    )
    if has_flag:
        return len(merge_filler_windows(temporal_data))
    if isinstance(temporal_data, list):
        return sum(1 for w in temporal_data if w.get("energy_status") == "low" and w.get("pitch_status") == "monotone")
    return 0


def analyze_speech_metrics(audio_path, model=None, model_size="medium", temporal_data=None):
    y, sr = librosa.load(audio_path, sr=None)
    total_audio_duration = len(y) / sr

    if model is None:
        model = whisper.load_model(model_size)

    result = model.transcribe(
        audio_path,
        fp16=False,
        word_timestamps=True,
        language="tr",
        temperature=0.0,
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6,
        initial_prompt=(
            "Bu bir Türkçe iş görüşmesidir. Konuşmacı zaman zaman "
            "'ıı', 'ıı ıı', 'ınınından', 'eee', 'ee', 'mmm' gibi dolgu sesleri çıkarıyor. "
            "Bu sesleri kelime gibi yaz, silme, düzeltme. "
            "Örnek: 'Benim adım ıı Ahmet. ınınından İstanbul da yaşıyorum. eee tecrübem var.'"
        )
    )

    # Speech time
    total_speech_time = 0.0
    all_segments_speech = 0.0
    for segment in result["segments"]:
        seg_dur = segment["end"] - segment["start"]
        all_segments_speech += seg_dur
        if segment.get("no_speech_prob", 0) > 0.6:
            continue
        total_speech_time += seg_dur

    if total_speech_time < 1.0:
        if temporal_data is not None:
            if isinstance(temporal_data, list) and len(temporal_data) > 0:
                silence_count = sum(1 for w in temporal_data if w.get("energy_status") == "silence")
            elif isinstance(temporal_data, pd.DataFrame) and not temporal_data.empty:
                silence_count = int((temporal_data["energy_status"] == "silence").sum())
            else:
                silence_count = 0
            total_speech_time = max(0.0, total_audio_duration - silence_count * 1.0)
        else:
            total_speech_time = all_segments_speech

    # Transcript
    text_full = result["text"]
    text_clean = _clean_transcript(text_full)
    filler_text, filler_debug = count_filler_words(text_full, FILLER_WORDS)
    total_words = len(re.sub(r'[^\w\s]', '', text_clean).split())

    # Pause
    total_pause_time = max(0.0, total_audio_duration - total_speech_time)
    pause_ratio = (total_pause_time / total_audio_duration) * 100 if total_audio_duration > 0 else 0.0

    window_sec = 1.0
    if isinstance(temporal_data, list) and len(temporal_data) > 0:
        silence_count = sum(1 for w in temporal_data if w.get("energy_status") == "silence")
        acoustic_pause_time = silence_count * window_sec
        acoustic_pause_ratio = (acoustic_pause_time / total_audio_duration) * 100 if total_audio_duration > 0 else 0.0
    elif isinstance(temporal_data, pd.DataFrame) and not temporal_data.empty:
        silence_count = int((temporal_data["energy_status"] == "silence").sum())
        acoustic_pause_time = silence_count * window_sec
        acoustic_pause_ratio = (acoustic_pause_time / total_audio_duration) * 100 if total_audio_duration > 0 else 0.0
    else:
        acoustic_pause_time = None
        acoustic_pause_ratio = pause_ratio

    if acoustic_pause_time is not None and total_audio_duration > 0:
        if abs(pause_ratio - acoustic_pause_ratio) > 15:
            total_speech_time = max(0.0, total_audio_duration - acoustic_pause_time)
            total_pause_time = max(0.0, total_audio_duration - total_speech_time)
            pause_ratio = (total_pause_time / total_audio_duration) * 100

    # =========================
    # Filler — Adaptif Hybrid
    # =========================
    # Senaryo 1: text=0, audio=0          → 0
    # Senaryo 2: text=0, audio<3          → 0  (hafif duraksamalar)
    # Senaryo 3: text=0, audio>=3         → audio*0.5  (Whisper silmiş)
    # Senaryo 4: text>0, audio=0          → text
    # Senaryo 5: div<=1.5                 → text*0.6 + audio*0.4
    # Senaryo 6: 1.5<div<=2.5             → text*0.8 + audio*0.2
    # Senaryo 7: div>2.5, text>0          → text
    # Senaryo 8: div>2.5, text=0          → audio*0.5
    AUDIO_MIN_EVENTS = 3

    filler_audio = estimate_fillers_from_audio(temporal_data) if temporal_data is not None else 0

    if filler_text == 0 and filler_audio == 0:
        filler_count = 0
        filler_source = "none"

    elif filler_text == 0 and filler_audio > 0:
        if filler_audio < AUDIO_MIN_EVENTS:
            filler_count = 0
            filler_source = "none(audio_noise_rejected)"
        else:
            filler_count = round(filler_audio * 0.5)
            filler_source = "audio_only(whisper_missed)"

    elif filler_text > 0 and filler_audio == 0:
        filler_count = filler_text
        filler_source = "text_only(audio_missed)"

    else:
        divergence = filler_audio / filler_text
        if divergence < 1.0:
            # Audio text'ten az buluyor → acoustic yetersiz kalmış, text güvenilir
            audio_w, text_w = 0.0, 1.0
            filler_source = "text_primary(audio_undercount)"
        elif divergence <= 1.5:
            audio_w, text_w = 0.4, 0.6
            filler_source = "hybrid_close"
        elif divergence <= 2.5:
            audio_w, text_w = 0.2, 0.8
            filler_source = "hybrid_text_dominant"
        else:
            if filler_text > 0:
                audio_w, text_w = 0.0, 1.0
                filler_source = "text_primary(audio_unreliable)"
            else:
                audio_w, text_w = 0.5, 0.0
                filler_source = "audio_halved(text_zero)"
        filler_count = round(filler_audio * audio_w + filler_text * text_w)

    filler_ratio = filler_count / max(total_words, 1)

    # WPM
    speech_minutes = max(total_speech_time / 60, 1e-6)
    wpm_gross = total_words / speech_minutes
    net_words = max(total_words - filler_text, 0)
    wpm_net = net_words / speech_minutes

    if wpm_gross < 110:
        speed_label = "slow"
    elif wpm_gross > 160:
        speed_label = "fast"
    else:
        speed_label = "normal"

    return {
        "total_words": total_words,
        "wpm": round(wpm_gross, 1),
        "wpm_net": round(wpm_net, 1),
        "pause_ratio": round(pause_ratio, 1),
        "acoustic_pause_ratio": round(acoustic_pause_ratio, 1),
        "filler_count": filler_count,
        "filler_ratio": round(filler_ratio, 4),
        "filler_source": filler_source,
        "filler_debug": filler_debug,
        "filler_text_raw": filler_text,
        "filler_audio_raw": filler_audio,
        "speed_label": speed_label,
        "transcript": text_full,
    }