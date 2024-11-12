import glob
import torch
import numpy as np
import os
import soundfile as sf
from collections import OrderedDict
import datetime
from tqdm import tqdm

def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sample_rate=22050,
):
    """
    Log various summaries to a TensorBoard writer.

    Args:
        writer (SummaryWriter): The TensorBoard writer.
        global_step (int): The current global step.
        scalars (dict, optional): Dictionary of scalar values to log.
        histograms (dict, optional): Dictionary of histogram values to log.
        images (dict, optional): Dictionary of image values to log.
        audios (dict, optional): Dictionary of audio values to log.
        audio_sample_rate (int, optional): Sampling rate of the audio data.
    """
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sample_rate)

def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    checkpoints = sorted(
        glob.glob(os.path.join(dir_path, regex)),
        key=lambda f: int("".join(filter(str.isdigit, f))),
    )
    return checkpoints[-1] if checkpoints else None

def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    checkpoint_data = {
        "model": state_dict,
        "iteration": iteration,
        "optimizer": optimizer.state_dict(),
        "learning_rate": learning_rate,
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Saved model '{checkpoint_path}' (epoch {iteration})")
    
def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
    assert os.path.isfile(checkpoint_path), f"Checkpoint file not found: {checkpoint_path}"
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model_state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    new_state_dict = {k: checkpoint_dict["model"].get(k, v) for k, v in model_state_dict.items()}
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    if optimizer and load_opt == 1:
        optimizer.load_state_dict(checkpoint_dict.get("optimizer", {}))
    print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint_dict['iteration']})")
    return model, optimizer, checkpoint_dict.get("learning_rate", 0), checkpoint_dict["iteration"]    

def save_model(ckpt, sr, pitch_guidance, name, model_dir, epoch, step, version, hps,):
    print(f"Saving model '{model_dir}' (epoch {epoch} and step {step})")

    pth_file = f"{name}_{epoch}e_{step}s.pth"

    opt = OrderedDict(
        weight={
            key: value.half() for key, value in ckpt.items() if "enc_q" not in key
        }
    )
    
    opt["config"] = [
        hps["data"]["filter_length"] // 2 + 1,
        32,
        hps["model"]["inter_channels"],
        hps["model"]["hidden_channels"],
        hps["model"]["filter_channels"],
        hps["model"]["n_heads"],
        hps["model"]["n_layers"],
        hps["model"]["kernel_size"],
        hps["model"]["p_dropout"],
        hps["model"]["resblock"],
        hps["model"]["resblock_kernel_sizes"],
        hps["model"]["resblock_dilation_sizes"],
        hps["model"]["upsample_rates"],
        hps["model"]["upsample_initial_channel"],
        hps["model"]["upsample_kernel_sizes"],
        hps["model"]["spk_embed_dim"],
        hps["model"]["gin_channels"],
        hps["data"]["sample_rate"],
    ]

    opt["epoch"] = epoch
    opt["step"] = step
    opt["sr"] = sr
    opt["f0"] = pitch_guidance
    opt["version"] = version
    opt["creation_date"] = datetime.datetime.now().isoformat()
    opt["model_name"] = name

    torch.save(opt, os.path.join(model_dir, pth_file))
    print(f"Model {pth_file} saved.")

def load_dataset(input_path):

    with open(os.path.join(input_path, "filelist.txt"), 'r') as f:
        files = [line.strip().split("|") for line in f]
    d = []
    
    with tqdm(total=len(files), leave=False) as pbar:
        for f in files:
            phone = np.load(f[1])
            phone = torch.FloatTensor(phone)
            phone_lengths = torch.LongTensor([50])

            pitch = np.load(f[3])
            pitch = torch.LongTensor(pitch)
        
            pitchf = np.load(f[2])
            pitchf = torch.FloatTensor(pitchf)
        
            spec = torch.load(f[4])
            #pad to 50 frames
            spec = torch.nn.functional.pad(spec, (0, 1), mode='constant', value=0)
            spec_lengths = torch.LongTensor([49])
        
            audio, sr = sf.read(f[0])
            wave = torch.FloatTensor(audio.astype(np.float32))
            wave = wave.unsqueeze(0)
            wave_lengths = torch.LongTensor([24000])
                
            sid = torch.LongTensor([int(f[5])])
        
            d.append(
                (   
                    phone,
                    phone_lengths,
                    pitch,
                    pitchf,
                    spec,
                    spec_lengths,
                    wave,
                    wave_lengths,
                    sid
                )
            )
            pbar.update(1)
    return d
