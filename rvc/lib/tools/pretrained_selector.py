def pretrained_selector(version, vocoder, pitch_guidance, sample_rate):
    
    path = f"rvc/models/pretraineds/pretrained_{version}/"
    f0 = "f0" if pitch_guidance == True else ""

    if vocoder == "default":
        vocoder_path = ""
    elif vocoder == "MRF HiFi-GAN":
        vocoder_path = "HiFiGAN_"

    path_g = f"{path}{vocoder_path}{f0}G{str(sample_rate)[:2]}k.pth"
    path_d = f"{path}{vocoder_path}{f0}D{str(sample_rate)[:2]}k.pth"

    return path_g, path_d