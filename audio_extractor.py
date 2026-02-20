import os
import librosa
import soundfile as sf
import tempfile

# MoviePy version safe import
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

SUPPORTED_FORMATS = (".mp4", ".mov", ".avi", ".mkv")


def extract_and_prep_audio(video_path, output_wav_path=None):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not video_path.lower().endswith(SUPPORTED_FORMATS):
        raise ValueError("Unsupported video format")

    if output_wav_path is None:
        output_wav_path = os.path.splitext(video_path)[0] + "_16k.wav"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_audio = tmp.name

    with VideoFileClip(video_path) as video:
        if video.audio is None:
            raise ValueError("Video has no audio track")

        video.audio.write_audiofile(temp_audio, logger=None)

    y, sr = librosa.load(temp_audio, sr=16000, mono=True)
    sf.write(output_wav_path, y, sr)

    os.remove(temp_audio)

    duration = len(y) / sr

    return {
        "audio_path": output_wav_path,
        "duration": duration,
        "sample_rate": sr,
        "channels": 1
    }