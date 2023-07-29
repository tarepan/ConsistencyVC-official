"""Wave-to-WEO preprocessing runner."""

import os
import argparse
from glob import glob

import numpy as np
import torch
import librosa
import soundfile as sf
from tqdm import tqdm

from whisper.model import Whisper, ModelDimensions
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram


def load_model(path) -> Whisper:
    """Load a pretrained Whisper model."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=device)
    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def wave_to_weo(whisper: Whisper, wavPath, p_out: str):

    # Hack for restriction, see issue#13 (https://github.com/ConsistencyVC/ConsistencyVC-voive-conversion/issues/13)
    audio, sr = librosa.load(wavPath, sr=None)
    if len(audio) >= sr * 29:
        print(wavPath, "cut to 29s")
        audio = audio[:sr * 29]
        sf.write(wavPath, audio, sr)

    # Load
    audio = load_audio(wavPath)

    with torch.no_grad():
        # Inference
        mel = log_mel_spectrogram(audio).unsqueeze(0).to(whisper.device)
        ppg = whisper.encoder(mel).squeeze().data.cpu().float().numpy()
        ppg = ppg[:audio.shape[0] // 320,] # [length, dim=1024]
        # Output
        np.save(p_out, ppg, allow_pickle=False)


if __name__ == "__main__":
    """
    Prerequisites:
        - .wav files under the `data_dir` directory
    """

    # List up all .wav files under the `data_dir`
    data_dir = r".\dataset\crosslingual_emo_dataset\jvs"
    wav_files = glob(os.path.join(data_dir, '**', '*.wav'), recursive=True)
    wav_files = sorted(wav_files)

    # Load the model
    whisper = load_model(os.path.join("whisper_pretrain", "medium.pt"))

    # Run preprocessing
    for p_wav in tqdm(wav_files):
        p_out = p_wav.replace(r".wav", r"whisper.pt.npy")
        if not os.path.exists(p_out):
            wave_to_weo(whisper, p_wav, p_out)
