# 本地视频理解

在项目目录中使用已经安装依赖的 `omnizip` 环境运行：

```powershell
conda activate omnizip
python app.py
```

也可以直接运行：

```powershell
conda run --no-capture-output -n omnizip python app.py
```

在 IDE 中运行时，将 Python 解释器设为 `C:\Users\wyz\.conda\envs\omnizip\python.exe`。
默认的 Anaconda base 环境没有安装 torch。

模型和视频路径在 `app.py` 中通过 `MODEL_PATH`、`VIDEO_PATH` 配置。
当前开启视频音频分析，输入需包含音轨；无音轨视频请将 `USE_AUDIO_IN_VIDEO` 设为 `False`。

本机 8 GB 显卡使用以下内存设置：加载时禁用语音生成、启用 SDPA、模型权重最多使用 5 GiB 显存，剩余权重自动放到 CPU；视频最多采样 16 帧，并限制每帧像素数。这会牺牲部分视觉细节，CPU 分担计算也会降低速度。更长的视频仍可能需要分段处理。

当前环境：PyTorch 2.6.0+cu124、Transformers 4.52.3。
项目里的 `modeling_qwen2_5_omni.py` 不会被 `app.py` 自动导入；实际使用的是环境中 Transformers 的模型实现。
