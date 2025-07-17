# Wan_Audio

采用Wan2.1 T2V 1.3B 作为基模进行开发。

依赖权重：
1. Wan2.1 T2V 1.3B: VAE,T5等权重和参数，可在 https://github.com/Wan-Video/Wan2.1 下载
2. whisper-tiny: 音频特征提取模型，可在 https://huggingface.co/openai/whisper-tiny 下载，默认路径为 `/mnt/data/public_ckpt/audio2feature/whisper-tiny`

实验设置：
1. `config/train_wan_a2v.yaml` 中可设置实验参数

train：
执行 `sh train.sh` 启动训练