import os
import subprocess
import numpy as np
import soundfile as sf
from scipy.signal import lfilter
import streamlit as st
from huggingface_hub import snapshot_download

# Forzar uso de CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# === EMOTION MODIFIER FUNCTION ===
def modify_audio_for_emotion(audio, sr, emotion):
    audio = audio.astype(np.float32)

    if emotion == 'neutral':
        return audio

    elif emotion == 'feliz':
        new_sr = int(sr * 1.03)
        audio_resampled = np.interp(np.linspace(0, len(audio) / sr, int(len(audio) * new_sr / sr)),
                                     np.linspace(0, len(audio) / sr, len(audio)),
                                     audio)
        delay_samples = int(0.03 * sr)
        decay = 0.4
        reverb = np.zeros_like(audio_resampled)
        reverb[delay_samples:] = audio_resampled[:-delay_samples] * decay
        final = audio_resampled + reverb
        return final / np.max(np.abs(final))

    elif emotion == 'triste':
        new_sr = int(sr * 0.97)
        audio_resampled = np.interp(np.linspace(0, len(audio) / sr, int(len(audio) * new_sr / sr)),
                                     np.linspace(0, len(audio) / sr, len(audio)),
                                     audio)
        b = np.ones(10) / 10.0
        filtered = lfilter(b, [1.0], audio_resampled)
        return filtered / np.max(np.abs(filtered))

    elif emotion == 'enojado':
        amplified = audio * 2.0
        distorted = np.tanh(amplified)
        return distorted / np.max(np.abs(distorted))

    return audio

# === TTS FUNCTION ===
@st.cache_resource
def download_model():
    REPO_ID = "kgemera/XTTSV2-es"
    return snapshot_download(repo_id=REPO_ID)

def generate_emotional_audio(text, emotion):
    base_output_path = "base_output.wav"
    final_output_path = "emotional_output.wav"

    model_dir = download_model()
    model_path = os.path.join(model_dir)
    config_path = os.path.join(model_dir, "config.json")
    speaker_wav = os.path.join(model_dir, "reference.wav")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"

    command_args = [
        "tts",
        "--model_path", model_path,
        "--config_path", config_path,
        "--text", text,
        "--speaker_wav", speaker_wav,
        "--language_idx", "es",
        "--out_path", base_output_path,
        "--use_cuda", "false"
    ]

    try:
        subprocess.run(command_args, check=True, capture_output=True, env=env)
        if not os.path.exists(base_output_path):
            return None
        audio, sr = sf.read(base_output_path)
        modified_audio = modify_audio_for_emotion(audio, sr, emotion)
        sf.write(final_output_path, modified_audio, sr)
        return final_output_path
    except subprocess.CalledProcessError as e:
        st.error("Error al ejecutar TTS:\n" + e.stderr.decode())
        return None
    except Exception as e:
        st.error("Error general: " + str(e))
        return None

# === Streamlit UI ===
st.set_page_config(page_title="Generador de Voz Emocional", layout="centered")
st.title("🎙️ Generador de Voz Emocional en Español")
st.write("Escribe un texto y selecciona una emoción para generar una voz emocional en español usando XTTSv2.")

text_input = st.text_area("Texto a sintetizar", placeholder="Escribe una frase...", height=150)
emotion = st.radio("Selecciona una emoción", ["neutral", "feliz", "triste", "enojado"])

if st.button("Generar voz"):
    if text_input.strip() == "":
        st.warning("Por favor, escribe un texto.")
    else:
        with st.spinner("Generando voz..."):
            audio_path = generate_emotional_audio(text_input, emotion)
        if audio_path:
            st.audio(audio_path)
        else:
            st.error("No se pudo generar el audio.")
