def run(input_path, pretrainD, pretrainG, batch_size, config, save_epochs):
    import rvc.lib.zluda
    import torch
    import torchaudio

    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    import numpy as np
    import soundfile as sf
    import os
    from random import shuffle
    from rvc.train.train_utils import (
        load_checkpoint,
        save_checkpoint,
        latest_checkpoint_path,
        load_dataset,
        save_model,
        summarize
    )
    from rvc.lib.algorithm import commons
    from rvc.train.losses import (discriminator_loss, feature_loss, generator_loss, kl_loss,)
    from accelerate import Accelerator
    from rvc.lib.algorithm.synthesizers import Synthesizer
    from rvc.lib.algorithm.discriminator import CombinedDiscriminator
    from rvc.lib.algorithm.mpd import MultiPeriodDiscriminator
    from rvc.lib.algorithm.msstftd import MultiScaleSTFTDiscriminator
    from rvc.lib.algorithm.mssbcqtd import MultiScaleSubbandCQTDiscriminator
    from rvc.train.mel_processing import MultiScaleMelSpectrogramLoss

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1234)
    
    print("Configuring models...")
    spec_channels = config["data"]["filter_length"] // 2 + 1
    segment_size = config["train"]["segment_size"] // config["data"]["hop_length"]
    
    net_g = Synthesizer(spec_channels, segment_size, **config["model"], use_f0=True, sr=config["data"]["sample_rate"], is_half=False, randomized=False)
   
    net_d = CombinedDiscriminator(
        [
            MultiPeriodDiscriminator(),
            MultiScaleSTFTDiscriminator(),
            MultiScaleSubbandCQTDiscriminator(sample_rate=config["data"]["sample_rate"])
        ]
    )

    optim_g = torch.optim.AdamW(
        net_g.parameters(), 
        config["train"]["learning_rate"], betas=config["train"]["betas"], eps=config["train"]["eps"],
    )

    optim_d = torch.optim.AdamW(
        net_d.parameters(), 
        config["train"]["learning_rate"], betas=config["train"]["betas"], eps=config["train"]["eps"],
    )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=config["train"]["lr_decay"], last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=config["train"]["lr_decay"], last_epoch=-1)

    fn_mel_loss_multiscale = MultiScaleMelSpectrogramLoss(sample_rate=config["data"]["sample_rate"])

    # accelerator wrapping
    accelerator = Accelerator()
    net_g, net_d, optim_g, optim_d, scheduler_g, scheduler_d, fn_mel_loss_multiscale = accelerator.prepare(net_g, net_d, optim_g, optim_d, scheduler_g, scheduler_d, fn_mel_loss_multiscale)
    device = accelerator.device

    #training data
    print("Loading training data...")

    data = load_dataset(input_path)

    print("Loading checkpoints...")
    try:
        _, _, _, epoch_str = load_checkpoint(latest_checkpoint_path(input_path, "D_*.pth"), net_d, optim_d)
        _, _, _, epoch_str = load_checkpoint(latest_checkpoint_path(input_path, "G_*.pth"), net_g, optim_g)
        epoch_str += 1
        global_step = int((epoch_str - 1) * len(data) // batch_size)
    except:
        epoch_str = 1
        global_step = 0
        if pretrainG != "" and pretrainG != "None" :
            if accelerator.is_main_process:
                print(f"Loaded pretrained (G) '{pretrainG}'")
                if hasattr(net_g, "module"):
                    net_g.module.load_state_dict(torch.load(pretrainG, map_location="cpu")["model"])
                else:
                    net_g.load_state_dict(torch.load(pretrainG, map_location="cpu")["model"])
    
        if pretrainD != "" and pretrainD != "None":
            if accelerator.is_main_process:
                print(f"Loaded pretrained (D) '{pretrainD}'")
            if hasattr(net_d, "module"):
                net_d.module.load_state_dict(torch.load(pretrainD, map_location="cpu")["model"])
            else:
                net_d.load_state_dict(torch.load(pretrainD, map_location="cpu")["model"])

    # tensorboard
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=input_path)
        writer_eval = SummaryWriter(log_dir=os.path.join(input_path, "eval"))

    print("Starting training...")

    for epoch in range(epoch_str, 101):
        shuffle(data)
        net_g.train()
        net_d.train()
    
        with tqdm(total=len(data)//batch_size, leave=False) as pbar:
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                #load batch tensors
                phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid = zip(*batch)
         
                phone           = torch.stack(phone).to(device, non_blocking=True)
                phone_lengths   = torch.cat(phone_lengths).to(device, non_blocking=True)
                pitch           = torch.stack(pitch).to(device, non_blocking=True)
                pitchf          = torch.stack(pitchf).to(device, non_blocking=True)
                spec            = torch.stack(spec).to(device, non_blocking=True)
                spec_lengths    = torch.cat(spec_lengths).to(device, non_blocking=True)
                wave            = torch.stack(wave).to(device, non_blocking=True)
                wave_lengths    = torch.cat(wave_lengths).to(device, non_blocking=True)
                sid             = torch.cat(sid).to(device, non_blocking=True)

                #print("\n")
                #print('phone         ', phone.shape)
                #print('phone_lengths ', phone_lengths.shape)
                #print(phone_lengths)
                #print('pitch         ', pitch.shape)
                #print('pitchf        ', pitchf.shape)
                #print('spec          ', spec.shape)
                #print('spec_lengths  ', spec_lengths.shape)
                #print(spec_lengths)
                #print('wave shape    ', wave.shape)
                #print('wave_lengths  ', wave_lengths.shape)
                #print(wave_lengths)
                #print('sid           ', sid.shape)
                #print(sid)
               
                model_output = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                y_hat, _, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = model_output
                #print('after gen')
                #print('wave shape', wave.shape)
                #print('y_hat', y_hat.shape)
                y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
                optim_d.zero_grad()
                accelerator.backward(loss_disc)
                grad_norm_d = commons.clip_grad_value(net_d.parameters(), None)
                optim_d.step()
                _, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
                #loss_mel = torch.nn.functional.l1_loss(wav_to_mel_spec(wave), wav_to_mel_spec(y_hat)) * config["train"]["c_mel"]
                loss_mel = fn_mel_loss_multiscale(wave, y_hat) * config["train"]["c_mel"]
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config["train"]["c_kl"]
                #if loss_kl.item() < 0:
                #    torch.save(z_p.detach(), 'z_p.pt')
                #    torch.save(logs_q.detach(), 'logs_q.pt')
                #    torch.save(m_p.detach(), 'm_p.pt')
                #    torch.save(logs_p.detach(), 'logs_p.pt')
                #    torch.save(z_mask.detach(), 'z_mask.pt')
               
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, _ = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
                optim_g.zero_grad()
                accelerator.backward(loss_gen_all)
                grad_norm_g = commons.clip_grad_value(net_g.parameters(), None)
                optim_g.step()
                with torch.no_grad():
                    print('\n')
                    #print(f">grad_norm_d={str(grad_norm_d)}, grad_norm_g={str(grad_norm_g)}, 
                    print(f"Epoch {epoch}: loss_gen_all={str(loss_gen_all.item())}, loss_gen={str(loss_gen.item())}, loss_disc={str(loss_disc.item())}")
                    print(f">loss_mel={str(loss_mel.item())}, loss_kl={str(loss_kl.item())}, loss_fm={str(loss_fm.item())}")
                global_step += 1                
                pbar.update(1)
            
        # end of epoch
        scheduler_g.step()
        scheduler_d.step()
    
        lr = optim_g.param_groups[0]["lr"]
        scalar_dict = {
            "loss/g/total": loss_gen_all,
            "loss/d/total": loss_disc,
            "learning_rate": lr,
            "grad/norm_d": grad_norm_d,
            "grad/norm_g": grad_norm_g,
            "loss/g/fm": loss_fm,
            "loss/g/mel": loss_mel,
            "loss/g/kl": loss_kl,
        }
        summarize(
            writer=writer,
            global_step=global_step,
            #images=image_dict,
            scalars=scalar_dict,
            #audios=audio_dict,
            #audio_sample_rate=config.data.sample_rate,
        )
    
        if epoch % save_epochs == 0:
            #torchaudio.save(f"outputs3\\y_hat_{epoch:06}.wav", y_hat[0].detach().cpu().squeeze(0), 48000)
            save_checkpoint(
                accelerator.unwrap_model(net_g), optim_g,
                config["train"]["learning_rate"], epoch, os.path.join(input_path, "G_" + str(epoch) + ".pth"),)
            save_checkpoint(
                accelerator.unwrap_model(net_d), optim_d,
                config["train"]["learning_rate"], epoch, os.path.join(input_path, "D_" + str(epoch) + ".pth"),)
        
            ckpt = accelerator.get_state_dict(net_g)
            save_model(
                ckpt=ckpt,
                sr=48000,
                pitch_guidance=True,
                name='GenTest',
                model_dir=input_path,
                epoch=str(epoch),
                step=str(global_step),
                version="v2",
                hps=config,
            )

if __name__ == "__main__":

    config = {
        "train": {
            "learning_rate": 0.0001, "betas": [0.8, 0.99], "eps": 1e-09, "lr_decay": 0.999875, "segment_size": 17280, "c_mel": 15.0, "c_kl": 1.0
        },
        "data": {
            "max_wav_value": 32768.0, "sample_rate": 48000, "filter_length": 2048, "hop_length": 480, "win_length": 2048, "n_mel_channels": 128, "mel_fmin": 0.0, "mel_fmax": None
        },
        "model": {
            "inter_channels": 192, "hidden_channels": 192, "filter_channels": 768, "text_enc_hidden_dim": 768, "n_heads": 2, "n_layers": 6, "kernel_size": 3, "p_dropout": 0,
            "resblock": "1","resblock_kernel_sizes": [3, 7, 11 ], "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            #"upsample_rates": [4, 4, 4, 2, 2], "upsample_kernel_sizes": [16, 12, 8, 6, 4], 
            "upsample_rates": [12, 10, 2, 2], "upsample_kernel_sizes": [24, 20, 4, 4], 
            "upsample_initial_channel": 512,
            "gin_channels": 256, "spk_embed_dim": 109, "use_spectral_norm": False,
        },
    }

    input_path = r"X:\GenTest_v2\input"
    pretrainD=r"" #X:\GenTest_v2\pretrained\D3_100.pth"
    pretrainG=r"" #X:\GenTest_v2\pretrained\G3_100.pth"
    batch_size = 4
    save_epochs = 1
    run(input_path, pretrainD, pretrainG, batch_size, config, save_epochs)