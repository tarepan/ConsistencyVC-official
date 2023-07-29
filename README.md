<div align="center">

# ConsistencyVC-official <!-- omit in toc -->
[![OpenInColab]][notebook]
[![paper_badge]][paper]

</div>

Clone of the official ***ConsistencyVC*** implementation.  
[Official demo](https://consistencyvc.github.io/ConsistencyVC-demo-page).  

- [`main` branch](https://github.com/tarepan/QuickVC-official/tree/main): Refactored & improved

<img src="cvc627.png" alt="cvc" width="100%">

## Pretrained models
- [whisper medium](https://drive.google.com/file/d/1PZsfQg3PUZuu1k6nHvavd6OcOB_8m1Aa/view?usp=drive_link)
- [XVC and EVC](https://drive.google.com/drive/folders/1KvMN1V8BWCzJd-N8hfyP283rLQBKIbig?usp=sharing)

## Usage
The audio needs to be 16KHz for train and inference.


### Inference
with the pre-trained models (use WEO as example)

#### XVC (WEO unit)
1. Generate the WEO of the source speech in [src](https://github.com/ConsistencyVC/ConsistencyVC-voive-conversion/blob/467ed5e632b2b328d01c87cb73e92b26b36deb05/whisperconvert_exp.py#L39C1-L39C1) by preprocess_ppg.py.
2. Copy the root of the reference speech to [tgt](https://github.com/ConsistencyVC/ConsistencyVC-voive-conversion/blob/467ed5e632b2b328d01c87cb73e92b26b36deb05/whisperconvert_exp.py#L47)
3. Use whisperconvert_exp.py to achieve voice conversion using WEO as content information.

#### EVC (PPG)

For ConsistencyEVC, use ppgemoconvert_exp.py to achieve voice conversion using ppg as content information.

### Train

#### XVC (WEO unit)
Use preprocess_ppg.py to generate the WEO.

First, train w/o speaker consistency loss for 100k steps:

change [this line](https://github.com/ConsistencyVC/ConsistencyVC-voive-conversion/blob/b5e8e984dffd5a12910d1846e25b128298933e40/train_whisper_emo.py#L214C11-L214C11) to 

```python
loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl# + loss_emo
```

run the py file:

```bash
python train_whisper_emo.py -c configs/cvc-whispers-multi.json     -m cvc-whispers-three
```

Then, finetune w/ speaker consistency loss:

change [this line](https://github.com/ConsistencyVC/ConsistencyVC-voive-conversion/blob/71cf17a5b65c12987ea7fba74d1d173ea1aae5cb/train_whisper_emo.py#L214) back to 

run the py file:

```bash
python train_whisper_emo.py -c configs/cvc-whispers-three-emo.json -m cvc-whispers-three
```

#### EVC (PPG)
Use ppg.py to generate the PPG.

First, train w/o speaker consistency loss for 100k steps:

change [this line](https://github.com/ConsistencyVC/ConsistencyVC-voive-conversion/blob/71cf17a5b65c12987ea7fba74d1d173ea1aae5cb/train_eng_ppg_emo_loss.py#L311) to 

```python
loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl# + loss_emo
```

run the py file:

```bash
python train_eng_ppg_emo_loss.py -c configs/cvc-eng-ppgs-three-emo.json -m cvc-eng-ppgs-three-emo
```

Then, finetune w/ speaker consistency loss:

change [this line](https://github.com/ConsistencyVC/ConsistencyVC-voive-conversion/blob/71cf17a5b65c12987ea7fba74d1d173ea1aae5cb/train_eng_ppg_emo_loss.py#L311) back to 

run the py file:

```bash
python train_eng_ppg_emo_loss.py -c configs/cvc-eng-ppgs-three-emo-cycleloss.json -m cvc-eng-ppgs-three-emo
```

## References

### Original paper <!-- omit in toc -->
[![paper_badge]][paper]  
<!-- Generated with the tool -> https://arxiv2bibtex.org/?q=2307.00393&format=bibtex -->
```bibtex
@misc{2307.00393,
Author = {Houjian Guo and Chaoran Liu and Carlos Toshinori Ishi and Hiroshi Ishiguro},
Title = {Using joint training speaker encoder with consistency loss to achieve cross-lingual voice conversion and expressive voice conversion},
Year = {2023},
Eprint = {arXiv:2307.00393},
}
```

### Acknowlegements

- [FreeVC-s](https://github.com/OlaWod/FreeVC) : code structure
- [LoraSVC](https://github.com/PlayVoice/lora-svc) : The WEO content feature
- [phoneme recognition model](https://huggingface.co/speech31/wav2vec2-large-english-TIMIT-phoneme_v3) : PPG


[paper]: https://arxiv.org/abs/2307.00393
[paper_badge]: http://img.shields.io/badge/paper-arxiv.2307.00393-B31B1B.svg
[notebook]: https://colab.research.google.com/github/tarepan/ConsistencyVC-official/blob/main/consistencyvc.ipynb
[OpenInColab]: https://colab.research.google.com/assets/colab-badge.svg