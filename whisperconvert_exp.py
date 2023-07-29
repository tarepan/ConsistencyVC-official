"""Inference runner."""

import os
import argparse
import time
import json
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
    parser.add_argument("--srctgtfile", type=str, default="./src_tgt.json",                                                       help="path to json src/tgt file")
    parser.add_argument("--hpfile",     type=str, default="./logs/cvc-whispers-three-emo-loss/config.json",                       help="path to json config file")
    parser.add_argument("--ptfile",     type=str, default="./logs/cvc-whispers-three-emo-loss/G_cvc-whispers-three-emo-loss.pth", help="path to pth file")
    parser.add_argument("--outdir",     type=str, default="./output/XVC",                                                         help="path to output dir")
    parser.add_argument("--device",     type=str, default="cuda",                                                                  help="inference device")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model & checkpoint...")
    net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, **hps.model).to(args.device)
    net_g.eval()
    utils.load_checkpoint(args.ptfile, net_g, None, True)

    # List up conversion pairs
    with open(args.srctgtfile) as f:
        src_tgt = json.load(f)
    src_wavs = src_tgt.srcs
    tgt_wavs = src_tgt.tgts
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
            wav_tgt = torch.from_numpy(wav_tgt_npy).unsqueeze(0).to(args.device)
            c = torch.from_numpy(np.load(c_filename)).transpose(1,0).unsqueeze(0).to(args.device)

            # Inference
            mel_tgt = mel_spectrogram_torch(wav_tgt,
                hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)
            audio = net_g.infer(c, mel=mel_tgt)

            # Output - Source/Target/Converted .wav files
            audio = audio[0][0].data.cpu().float().numpy()
            sf.write(os.path.join(args.outdir, f"{srcname}.wav"), wav_src_npy, hps.data.sampling_rate)
            sf.write(os.path.join(args.outdir, f"{tgtname}.wav"), wav_tgt_npy, hps.data.sampling_rate)
            write(   os.path.join(args.outdir,   f"{title}.wav"),              hps.data.sampling_rate, audio)
