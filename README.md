# ConsistencyVC-voive-conversion

Demo page: https://consistencyvc.github.io/ConsistencyVC-demo-page

The whisper medium model can be downloaded here: https://drive.google.com/file/d/1PZsfQg3PUZuu1k6nHvavd6OcOB_8m1Aa/view?usp=drive_link

The pre-trained models are available here:https://drive.google.com/drive/folders/1KvMN1V8BWCzJd-N8hfyP283rLQBKIbig?usp=sharing


<!-- 科研好累。 -->

# Inference with the pre-trained models

Use whisperconvert_exp.py to achieve voice conversion using weo as content information.

Use ppgemoconvert_exp.py to achieve voice conversion using ppg as content information.

# Train models by your dataset

Use ppg.py to generate the PPG.

Use preprocess_ppg.py to generate the WEO.

If you want to use WEO to train a cross-lingual voice conversion model:

First you need to train the model without speaker consistency loss for 100k steps:

change [this line](https://github.com/ConsistencyVC/ConsistencyVC-voive-conversion/blob/b5e8e984dffd5a12910d1846e25b128298933e40/train_whisper_emo.py#L214C11-L214C11) to 

```python
loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl# + loss_emo
```

run the py file:

python train_whisper_emo.py -c configs/freevc.json -m freevc -->

Then change back to finetune this model with speaker consistency loss
If you want to use PPG to train an expressive voice conversion model:
<!-- python train.py -c configs/freevc.json -m freevc cvc-eng-ppgs-three-emo-cycleloss.json
cvc-eng-ppgs-three-emo.json
cvc-whispers-multi.json
cvc-whispers-three-emo.json-->

# Reference

The code structure is based on [FreeVC-s](https://github.com/OlaWod/FreeVC)

The WEO content feature is based on [LoraSVC](https://github.com/PlayVoice/lora-svc)

The PPG is from the [phoneme recognition model](https://huggingface.co/speech31/wav2vec2-large-english-TIMIT-phoneme_v3)
