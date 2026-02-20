from audio_extractor import extract_and_prep_audio
from speech_analyzer import analyze_speech_metrics
from acoustic_analyzer import analyze_temporal_acoustics


def run_full_pipeline(video_path, whisper_model=None):
    # 1. Ses çıkar
    audio_meta = extract_and_prep_audio(video_path)
    audio_path = audio_meta["audio_path"]

    # 2. Akustik analiz önce çalışır
    acoustic_result = analyze_temporal_acoustics(audio_path)
    temporal_data = acoustic_result["temporal_data"]   # list of dicts
    emphasis_events = acoustic_result["emphasis_events"]
    filler_events = acoustic_result["filler_events"]
    filler_duration = acoustic_result["filler_duration"]

    # 3. Konuşma analizi — temporal_data hibrit filler tespiti için kullanılır
    speech_data = analyze_speech_metrics(
        audio_path,
        model=whisper_model,
        temporal_data=temporal_data
    )

    return {
        **speech_data,
        "temporal_data": temporal_data,
        "emphasis_events": emphasis_events,
        "filler_events": filler_events,
        "filler_duration": filler_duration,
        "audio_meta": audio_meta
    }