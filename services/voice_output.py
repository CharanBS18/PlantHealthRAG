import os
import subprocess
import tempfile

from config import ENABLE_TTS


def _read_audio(path: str) -> bytes | None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, "rb") as audio_file:
        return audio_file.read()


def _is_valid_wav(data: bytes | None) -> bool:
    if not data or len(data) < 44:
        return False
    return data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _macos_say_wav(text: str) -> bytes | None:
    aiff_fd, aiff_path = tempfile.mkstemp(suffix=".aiff", prefix="plant_health_")
    os.close(aiff_fd)
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="plant_health_")
    os.close(wav_fd)
    try:
        say_completed = subprocess.run(
            ["say", "-o", aiff_path, text],
            check=False,
            capture_output=True,
        )
        if say_completed.returncode != 0:
            return None

        convert_completed = subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@22050", aiff_path, wav_path],
            check=False,
            capture_output=True,
        )
        if convert_completed.returncode != 0:
            return None

        wav_bytes = _read_audio(wav_path)
        return wav_bytes if _is_valid_wav(wav_bytes) else None
    finally:
        if os.path.exists(aiff_path):
            os.remove(aiff_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)


def synthesize_speech(text: str) -> tuple[bytes, str] | None:
    if not ENABLE_TTS or not text.strip():
        return None

    # First choice: pyttsx3 WAV output.
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="plant_health_")
    os.close(wav_fd)
    try:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.save_to_file(text, wav_path)
            engine.runAndWait()
            wav_bytes = _read_audio(wav_path)
            if _is_valid_wav(wav_bytes):
                return wav_bytes, "audio/wav"
        except Exception:
            pass

        # macOS fallback: system `say` + `afconvert` to valid WAV.
        wav_bytes = _macos_say_wav(text)
        if _is_valid_wav(wav_bytes):
            return wav_bytes, "audio/wav"

        return None
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
