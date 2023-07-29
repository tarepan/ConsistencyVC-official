"""Inference runner."""

import os
import argparse
import time
import logging
logging.getLogger('numba').setLevel(logging.WARNING)

import torch
import librosa
import numpy as np
from scipy.io.wavfile import write
from tqdm import tqdm
import soundfile as sf

import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch


if __name__ == "__main__":
    """
    Prerequisites:
        - Preprocessed WEO in the same directories
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, default="./logs/cvc-whispers-three-emo-loss/config.json",                       help="path to json config file")
    parser.add_argument("--ptfile", type=str, default="./logs/cvc-whispers-three-emo-loss/G_cvc-whispers-three-emo-loss.pth", help="path to pth file")
    parser.add_argument("--outdir", type=str, default="output/60_exp_crosslingual_whispers-three-emo-loss",                   help="path to output dir")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model & checkpoint...")
    net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, **hps.model).cuda()
    net_g.eval()
    utils.load_checkpoint(args.ptfile, net_g, None, True)

    # List up conversion pairs
    src_wavs=[r".\dataset\crosslingual_emo_dataset\LibriTTS100\911\128684\911_128684_000004_000001.wav",
             r".\dataset\crosslingual_emo_dataset\LibriTTS100\730\359\730_359_000004_000001.wav",
             r".\dataset\crosslingual_emo_dataset\aishell3\wav\SSB0246\SSB02460001.wav",
             r".\dataset\crosslingual_emo_dataset\aishell3\wav\SSB1863\SSB18630001.wav",
             r".\dataset\crosslingual_emo_dataset\jvs\jvs003\nonpara30\wav24kHz16bit\BASIC5000_0440.wav",
             r".\dataset\crosslingual_emo_dataset\jvs\jvs014\nonpara30\wav24kHz16bit\BASIC5000_0318.wav"]
    tgt_wavs=[r".\dataset\crosslingual_emo_dataset\LibriTTS100\27\123349\27_123349_000003_000002.wav",
             r".\dataset\crosslingual_emo_dataset\LibriTTS100\87\121553\87_121553_000254_000000.wav",
             r".\dataset\crosslingual_emo_dataset\aishell3\wav\SSB1935\SSB19350001.wav",
             r".\dataset\crosslingual_emo_dataset\aishell3\wav\SSB1759\SSB17590008.wav",
             r".\dataset\crosslingual_emo_dataset\jvs\jvs009\nonpara30\wav24kHz16bit\BASIC5000_0155.wav",
             r".\dataset\crosslingual_emo_dataset\jvs\jvs010\nonpara30\wav24kHz16bit\BASIC5000_0113.wav",
             r".\dataset\vctk-16k\p304\p304_007.wav",
             r".\jecs_ref\JECS0000_JA.wav",
             r".\aishell1_ref\BAC009S0655W0493.wav"]
    titles: list[str] = [] # Conversion names
    srcs:   list[str] = [] # Source audio file paths
    tgts:   list[str] = [] # Target audio file paths
    for src_wav in src_wavs:
        for tgt_wav in tgt_wavs:
            src_wav_name = os.path.basename(src_wav)[:-4]
            tgt_wav_name = os.path.basename(tgt_wav)[:-4]
            title = f"{src_wav_name}_to_{tgt_wav_name}"
            titles.append(title)
            srcs.append(src_wav)
            tgts.append(tgt_wav)

    print("Synthesizing...")
    with torch.no_grad():
        for title, src, tgt in tqdm(zip(titles, srcs, tgts)):

            # Names
            srcname, tgtname = title.split("to")
            c_filename = src.replace(".wav", "whisper.pt.npy")

            # Load
            wav_src_npy = librosa.load(src, sr=hps.data.sampling_rate)[0]
            wav_tgt_npy = librosa.load(tgt, sr=hps.data.sampling_rate)[0]
            wav_tgt = torch.from_numpy(wav_tgt_npy).unsqueeze(0).cuda()
            c = torch.from_numpy(np.load(c_filename)).transpose(1,0).unsqueeze(0).cuda()

            # Inference
            mel_tgt = mel_spectrogram_torch(wav_tgt,
                hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)
            audio = net_g.infer(c, mel=mel_tgt)

            # Output - Source/Target/Converted .wav files
            audio = audio[0][0].data.cpu().float().numpy()
            sf.write(os.path.join(args.outdir, f"{srcname}.wav"), wav_src_npy, hps.data.sampling_rate)
            sf.write(os.path.join(args.outdir, f"{tgtname}.wav"), wav_tgt_npy, hps.data.sampling_rate)
            write(   os.path.join(args.outdir,   f"{title}.wav"),              hps.data.sampling_rate, audio)
