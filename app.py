try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise SystemExit(
        "当前 Python 环境未安装 torch。请先运行 conda activate omnizip，"
        "再运行 python app.py；或使用 conda run --no-capture-output -n omnizip python app.py。"
    ) from exc
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)
from qwen_omni_utils import process_mm_info

# 本地模型目录
MODEL_PATH = r"D:\Models\Qwen2.5-Omni-3B"

# 输入视频
VIDEO_PATH = r"D:\Videos\example.mp4"

# 模型加载
print("Loading model...")

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    # 为视频特征和生成过程预留显存（本机为 8 GB 显卡）。
    max_memory={0: "5GiB", "cpu": "24GiB"} if torch.cuda.is_available() else None,
    attn_implementation="sdpa",
    enable_audio_output=False,
    local_files_only=True,
)

# 不生成语音，只输出文字，减少显存占用
model.disable_talker()

processor = Qwen2_5OmniProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

print("Model loaded.")

# 构造对话
conversation = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a helpful assistant."
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": VIDEO_PATH,
                # 限制视频输入规模，避免旧版视觉注意力耗尽 8 GB 显存。
                "max_pixels": 144 * 28 * 28,
                "max_frames": 16,
            },
            {
                "type": "text",
                "text": "请用中文详细描述这个视频的内容。"
            },
        ],
    },
]

# 视频包含音频时，同时分析视频和音频
USE_AUDIO_IN_VIDEO = True

print("Processing video...")

text = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=False,
)

audios, images, videos = process_mm_info(
    conversation,
    use_audio_in_video=USE_AUDIO_IN_VIDEO,
)

inputs = processor(
    text=text,
    audio=audios,
    images=images,
    videos=videos,
    return_tensors="pt",
    padding=True,
    use_audio_in_video=USE_AUDIO_IN_VIDEO,
)

# 将输入放到模型所在设备
inputs = inputs.to(model.device).to(model.dtype)

print("Generating response...")

with torch.inference_mode():
    text_ids = model.generate(
        **inputs,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
        return_audio=False,
        max_new_tokens=256,
    )

# 解码模型输出
response = processor.batch_decode(
    text_ids[:, inputs.input_ids.shape[1]:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print("\n===== Model Response =====")
print(response[0])
