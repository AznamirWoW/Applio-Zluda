import os
import glob
import librosa
import soundfile as sf
import numpy as np
import torch
from torch import nn
from scipy import signal
from scipy.io import wavfile
from rvc.lib.predictors.RMVPE import RMVPE0Predictor
from transformers import HubertModel
from tqdm import tqdm

def run_split(input_path, output_path, sr):
    audio, sr = librosa.load(input_path, sr=sr)
    #apply a filter
    b_high, a_high = signal.butter(N=5, Wn=48, btype="high", fs=sr)
    audio = signal.lfilter(b_high, a_high, audio)
    chunk_length = int(sr * 0.5)
    with tqdm(total=len(audio)//chunk_length, leave=False) as pbar:
        for i in range(0, len(audio), chunk_length):
            chunk = audio[i:i + chunk_length]
            wavfile.write(os.path.join(output_path, f"chunk_{i//chunk_length:04}.wav"), sr, chunk.astype(np.float32))
            pbar.update(1)
          
def run_resample(files):
    for f in files:
        audio, sr = sf.read(f[0])
        audio_16k = librosa.resample(audio, orig_sr = sr, target_sr = 16000)
        
        wavfile.write(f[1], sr,    audio.astype(np.float32))
        wavfile.write(f[2], 16000, audio_16k.astype(np.float32))

def run_f0_extract(files):
    def cf0(f0):
        f0_bin = 256
        f0_max = 1100.0
        f0_min = 50.0
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        """Convert F0 to coarse F0."""
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel = np.clip((f0_mel - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1, 1, f0_bin - 1,)
        return np.rint(f0_mel).astype(int)

    model_rmvpe = RMVPE0Predictor(
                os.path.join("rvc", "models", "predictors", "rmvpe.pt"),
                is_half=False,
                device="cpu",)
    with tqdm(total=len(files), leave=False) as pbar:
        for f in files:
            audio, sr = sf.read(f[2]) # 16k file
            fine_f0 = model_rmvpe.infer_from_audio(audio, thred=0.03)
            fine_f0 = fine_f0[:-1] # removed extra frame
            coarse_f0 = cf0(fine_f0)
            np.save(f[3], fine_f0, allow_pickle=False)
            np.save(f[4], coarse_f0, allow_pickle=False)
            pbar.update(1)

def run_feats_extract(files):
    class HubertModelWithFinalProj(HubertModel):
        def __init__(self, config):
            super().__init__(config)
            self.final_proj = nn.Linear(768, config.classifier_proj_size)

    model_path = r"X:\Applio\rvc\models\embedders\contentvec"
    model = HubertModelWithFinalProj.from_pretrained(model_path)
    model.eval()
    with tqdm(total=len(files), leave=False) as pbar:
        for f in files:
            audio, sr = sf.read(f[2]) # 16k file
            feats = torch.from_numpy(audio).to(torch.float32).to("cpu")
            feats = torch.nn.functional.pad(feats.unsqueeze(0), (40,40), mode='reflect')
            feats = feats.view(1, -1)
            with torch.no_grad():
                feats = model(feats)["last_hidden_state"]
                feats = feats.squeeze(0).float().cpu().numpy()
            feats = np.repeat(feats, 2, axis=0) #expanded to match f0 shape
            np.save(f[5], feats, allow_pickle=False)
            pbar.update(1)

def run_spec_extract(files):
    def spectrogram_torch(y, n_fft, hop_size, win_size):
        hann_window= torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

        y = y.squeeze(1)
        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window,
            return_complex=True,
        )
        
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)
        return spec
    with tqdm(total=len(files), leave=False) as pbar:
        for f in files:
            audio, sample_rate = sf.read(f[1])
            audio = torch.FloatTensor(audio.astype(np.float32))

            audio_norm = audio.unsqueeze(0)
            spec = spectrogram_torch(
                audio_norm,
                n_fft=2048,
                hop_size=480,
                win_size=2048,
            )
            spec = spec[:,:,1:-1] # remove padding
            spec = torch.squeeze(spec, 0)
            torch.save(spec, f[6], _use_new_zipfile_serialization=False)
            pbar.update(1)
   
def run_filelist(files, save_path):
    
    with open(os.path.join(save_path, "filelist.txt"), "w") as filelist:
        for f in files:
            output= f"{f[1]}|{f[5]}|{f[3]}|{f[4]}|{f[6]}|0"
            filelist.write(output+"\n")
   
if __name__ == "__main__":
    save_path = r"X:\GenTest_v2\input"
 
    run_split(input_path=r"X:\GenTest_v2\training\Book_01.wav",  output_path=os.path.join(save_path, 'chunks'), sr=48000,)
    print('Split done')

    files = []
    for file in glob.glob(os.path.join(save_path, "chunks", "*.wav")):
        file_name = os.path.basename(file)
        file_info = [
            file,
            os.path.join(save_path, "sliced_audios", file_name),
            os.path.join(save_path, "sliced_audios_16k", file_name),
            os.path.join(save_path, "f0", file_name.replace("wav", "npy")),
            os.path.join(save_path, "f0c", file_name.replace("wav", "npy")),
            os.path.join(save_path, "feats", file_name.replace("wav", "npy")),
            os.path.join(save_path, "spec", file_name.replace("wav", "pt")),
        ]
        files.append(file_info)

    run_resample(files)
    print('Resample done')
    run_f0_extract(files)
    print('F0 done')
    run_feats_extract(files)
    print('Feats done')
    run_spec_extract(files)
    print('Spec done')
    run_filelist(files, save_path)