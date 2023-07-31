"""Train the ConsistencyVC XVC model."""

import os
import json
import argparse
import itertools
import math

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

import commons
import utils
from data_utils_whisper import TextAudioSpeakerLoader, TextAudioSpeakerCollate, DistributedBucketSampler
from models import SynthesizerTrn, MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch


torch.backends.cudnn.benchmark = True
global_step = 0


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."
  hps = utils.get_hparams()

  print("start run")
  run(hps)


def run(hps):
  global global_step

  # Init
  assert hps.model.use_spk is False, "use_spk is deprecated."
  logger = utils.get_logger(hps.model_dir)
  logger.info(hps)
  utils.check_git_hash(hps.model_dir)
  writer      = SummaryWriter(log_dir=hps.model_dir)
  writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
  torch.manual_seed(hps.train.seed)

  # Data
  train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps)
  train_sampler = DistributedBucketSampler(train_dataset, hps.train.batch_size,
      [75,100,125,150,175,200,225,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1100,1200,1300,1400,1500,2000,3000,4000,5000],
      num_replicas=1, rank=0, shuffle=True)
  collate_fn = TextAudioSpeakerCollate(hps)
  train_loader = DataLoader(train_dataset, num_workers=12, shuffle=False, pin_memory=True, collate_fn=collate_fn, batch_sampler=train_sampler)
  eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps)
  eval_loader = DataLoader(eval_dataset, num_workers=2, shuffle=True, batch_size=hps.train.batch_size, pin_memory=False, drop_last=False, collate_fn=collate_fn)

  # Models
  net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, **hps.model).cuda()
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda()
  optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
  optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
  try:
    epoch_str, global_step = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    epoch_str, global_step = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
    assert global_step is not None, "New training must support `global_step` checkpoint."
  except:
    epoch_str = 1
    global_step = 0
  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  # Trainer
  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    train_and_evaluate(epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    scheduler_g.step()
    scheduler_d.step()

    # Training finish
    if global_step > hps.train.max_global_step:
      utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, f"G_{global_step}.pth"), global_step)
      utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, f"D_{global_step}.pth"), global_step)
      print(f"global_step={global_step} (>{hps.train.max_global_step}), finished.")
      return


def train_and_evaluate(epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  
  net_g,        net_d       = nets
  optim_g,      optim_d     = optims
  scheduler_g,  scheduler_d = schedulers
  train_loader, eval_loader = loaders
  writer,       writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  net_g.train()
  net_d.train()

  for batch_idx, (c, spec, y) in enumerate(train_loader):
    # ==== step =========================================================================================================================================

    # Data
    c, spec, y = c.cuda(non_blocking=True), spec.cuda(non_blocking=True), y.cuda(non_blocking=True)

    # Transform
    mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)

    with autocast(enabled=hps.train.fp16_run):
      # Common_Forward
      y_hat, ids_slice, z_mask, (_, z_p, m_p, logs_p, _, logs_q), emo_y = net_g(c, spec, g=None, mel=mel)
      y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1),
          hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)

      # D_Forward
      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      # D_Loss
      with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
    # D_Backward/Optim
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with autocast(enabled=hps.train.fp16_run):
      # G_Forward
      _, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
      # Diff from QuickVC-official: emo
      emo_y_hat = net_g.enc_spk(y_hat_mel.transpose(1,2))
      # G_Loss
      with autocast(enabled=False):
        # Diff from QuickVC-official:
        #   - `0.5 * ` in loss_fm
        #   - loss_emo
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_fm  = 0.5                   * feature_loss(fmap_r, fmap_g)
        loss_mel = hps.train.c_mel       * F.l1_loss(y_hat_mel, y_mel)
        loss_kl  = hps.train.c_kl        * kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
        loss_emo = 0.5 * hps.train.c_mel * F.l1_loss(emo_y_hat, emo_y)
        if global_step < 100_000:
          loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        else:
          loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_emo
    # G_Backward/Optim
    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    # Train logging
    if global_step % hps.train.log_interval == 0:
      lr = optim_g.param_groups[0]['lr']
      losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl,loss_emo]
      logger.info('Train Epoch: {} [{:.0f}%]'.format(epoch, 100. * batch_idx / len(train_loader)))
      logger.info([x.item() for x in losses] + [global_step, lr])
      
      scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
      scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl, "loss/g/emo": loss_emo})

      scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
      scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
      scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
      image_dict = { 
          "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
          "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
          "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
      }
      utils.summarize(writer=writer, global_step=global_step, images=image_dict, scalars=scalar_dict)

    if global_step % hps.train.eval_interval == 0:
      # Evaluation
      evaluate(hps, net_g, eval_loader, writer_eval)
      # Checkpointing
      utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, f"G_{global_step}.pth"), global_step)
      utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, f"D_{global_step}.pth"), global_step)

    global_step += 1
    # ==== /step ========================================================================================================================================

  logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, generator, eval_loader, writer_eval):
    """NOTE: Totally (semantically) same as QuickVC-official."""
    generator.eval()
    with torch.no_grad():
      # Data
      for c, spec, y in eval_loader:
        c, spec, y = c[:1].cuda(), spec[:1].cuda(), y[:1].cuda()
        break
      # Transform
      mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
      y_hat = generator.infer(c, g=None, mel=mel)
      y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1).float(),
        hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
      "gt/mel":  utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
    }
    audio_dict = { "gen/audio": y_hat[0], "gt/audio": y[0], }
    utils.summarize(writer=writer_eval, global_step=global_step, images=image_dict, audios=audio_dict, audio_sampling_rate=hps.data.sampling_rate)
    generator.train()

                           
if __name__ == "__main__":
  main()
