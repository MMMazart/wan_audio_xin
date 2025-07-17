# pylint: disable=C0301
'''
This module contains the AudioProcessor class and related functions for processing audio data.
It utilizes various libraries and models to perform tasks such as preprocessing, feature extraction,
and audio separation. The class is initialized with configuration parameters and can process
audio files using the provided models.
'''
import math
import os

import librosa
import numpy as np
import torch
# from audio_separator.separator import Separator
from einops import rearrange
from transformers import WhisperModel, CLIPVisionModelWithProjection, AutoFeatureExtractor

class AudioProcessor:
    """
    AudioProcessor is a class that handles the processing of audio files.
    It takes care of preprocessing the audio files, extracting features
    using wav2vec models, and separating audio signals if needed.

    :param sample_rate: Sampling rate of the audio file
    :param fps: Frames per second for the extracted features
    :param wav2vec_model_path: Path to the wav2vec model
    :param only_last_features: Whether to only use the last features
    :param audio_separator_model_path: Path to the audio separator model
    :param audio_separator_model_name: Name of the audio separator model
    :param cache_dir: Directory to cache the intermediate results
    :param device: Device to run the processing on
    """
    def __init__(
        self,
        whisper_path,
        sample_rate=16000,
        device="cuda:0",
    ) -> None:
        self.sample_rate = sample_rate
        self.device = device

        self.audio_encoder = WhisperModel.from_pretrained(whisper_path).to(device).eval()
        self.audio_encoder.requires_grad_(False)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_path)

    def preprocess(self, wav_file: str):
        """
        Preprocess a WAV audio file by separating the vocals from the background and resampling it to a 16 kHz sample rate.
        The separated vocal track is then converted into wav2vec2 for further processing or analysis.

        Args:
            wav_file (str): The path to the WAV file to be processed. This file should be accessible and in WAV format.

        Raises:
            RuntimeError: Raises an exception if the WAV file cannot be processed. This could be due to issues
                        such as file not found, unsupported file format, or errors during the audio processing steps.

        Returns:
            torch.tensor: Returns an audio embedding as a torch.tensor
        """
        audio_input, sampling_rate = librosa.load(wav_file, sr=self.sample_rate)
        assert sampling_rate == self.sample_rate

        audio_len = len(audio_input) // 640

        audio_features = []
        window = 750*640 # whisper-特征提取模块接受的输入大小是固定30s，超长的音频要做裁剪然后拼接
        for i in range(0, len(audio_input), window):
            audio_feature = self.feature_extractor(audio_input[i:i+window], 
                                            sampling_rate=sampling_rate, 
                                            return_tensors="pt", 
                                            ).input_features
            audio_features.append(audio_feature)
        audio_features = torch.cat(audio_features, dim=-1).to(self.device)

        window = 3000 # 同样的wishper的模型输入也是固定长度3000的
        audio_prompts = []
        last_audio_prompts = []
        for i in range(0, audio_features.shape[-1], window):
            audio_prompt = self.audio_encoder.encoder(audio_features[:,:,i:i+window], output_hidden_states=True).hidden_states
            last_audio_prompt = self.audio_encoder.encoder(audio_features[:,:,i:i+window]).last_hidden_state
            last_audio_prompt = last_audio_prompt.unsqueeze(-2)
            audio_prompt = torch.stack(audio_prompt, dim=2)
            audio_prompts.append(audio_prompt)
            last_audio_prompts.append(last_audio_prompt)
        # 超过长度的做切片和拼接
        audio_prompts = torch.cat(audio_prompts, dim=1)
        # 这里按照实际的音频长度做了截断，一帧对应两位
        audio_prompts = audio_prompts[:,:audio_len*2]
        last_audio_prompts = torch.cat(last_audio_prompts, dim=1)
        last_audio_prompts = last_audio_prompts[:,:audio_len*2]

        return {'audio_prompts': audio_prompts.cpu(), 'last_audio_prompts':last_audio_prompts.cpu()}

    def close(self):
        """
        TODO: to be implemented
        """
        return self

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
