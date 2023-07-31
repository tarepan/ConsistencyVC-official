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
    dims = ModelDimensions(**checkpoint["dims"]) # Parameters
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def wave_to_weo(whisper: Whisper, wavPath: str, p_out: str):
    """Convert a waveform into Whisper Encoder Output (WEO) unit series.

    WEO :: (Frame=t/320, Feat=) - Whisper Encoder Output, hop=320 (20 msec)
    """

    # Hack for restriction, see issue#13 (https://github.com/ConsistencyVC/ConsistencyVC-voive-conversion/issues/13)
    audio, sr = librosa.load(wavPath, sr=None)
    if len(audio) >= sr * 29:
        print(wavPath, "cut to 29s")
        audio = audio[:sr * 29]
        sf.write(wavPath, audio, sr)

    # Load
    audio = load_audio(wavPath)

    # Inference
    with torch.no_grad():
        # wave2mel :: (T=t,) -> -> (Freq=80, Frame=t/160) -> (B=1, Freq=80, Frame=t/160)
        mel = log_mel_spectrogram(audio).unsqueeze(0).to(whisper.device)
        # mel2unit :: (B=1, Freq=80, Frame=t/160) -> (B=1, Frame=t/320, Feat) -> (Frame=t/320, Feat)
        unit = whisper.encoder(mel).squeeze().data.cpu().float().numpy()
        unit = unit[:audio.shape[0] // 320,] # [length, dim=1024]

    # Output
    np.save(p_out, unit, allow_pickle=False)


if __name__ == "__main__":
    """
    Prerequisites:
        - .wav files under the `data_dir` directory
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--audiodir", type=str, default="./data", help="Audio directory")
    parser.add_argument("--model",    type=str, default="./consistencyvc_official/medium.pt", help="Audio directory")
    args = parser.parse_args()
    
    # List up all .wav files under the `data_dir`
    wav_files = glob(os.path.join(args.audiodir, '**', '*.wav'), recursive=True)
    wav_files = sorted(wav_files)

    # Load the model
    whisper = load_model(args.model)

    # Run preprocessing
    for p_wav in tqdm(wav_files):
        p_out = p_wav.replace(r".wav", r"whisper.pt.npy")
        if not os.path.exists(p_out):
            wave_to_weo(whisper, p_wav, p_out)
