# Qwen2.5-Omni 本地 WorldSense 评测

默认从 `D:\Datasets\WorldSense` 读取视频和选择题，模型路径为 `D:\Models\Qwen2.5-Omni-3B`。

## 启动

### 安装环境

推荐 Python 3.10，当前本机版本为 3.10.21。在项目根目录执行：

```bash
conda create -n worldsense python=3.10 -y
conda activate worldsense
python -m pip install --upgrade pip
# 与本机一致的 CUDA 12.4 PyTorch 构建；需要兼容的 NVIDIA 驱动
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirement.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python app.py --check-data
```

`requirement.txt` 固定项目核心依赖的本机版本，包含可选 4-bit 功能所需的 bitsandbytes；不是完整传递依赖锁文件。没有加入本项目不使用的 lmms-eval、训练库和开发工具。服务器需修改 config.json 的模型、数据路径，确保安装版本支持服务器 GPU；以上 CUDA 组合不保证支持所有新型号 GPU。

当前 qwen-omni-utils 是本地修改版（含 PyAV 音频读取逻辑），已将源码快照保存在 `vendor/qwen-omni-utils`，安装清单直接引用它，迁移时必须一并上传 vendor 目录。标准 PyPI 的同版本不能保证具有相同行为。项目内模型文件同样需要上传，且依赖 Transformers 4.52.3 内部接口，不建议单独升级。

Linux 缺少媒体或声音系统库时可安装 `ffmpeg`、`libsndfile1`（Ubuntu/Debian：`sudo apt-get install ffmpeg libsndfile1`）；NVIDIA 驱动和这些系统组件不属于 pip 安装清单。当前整理依据是已验证的 Windows 环境，未在全新 Linux 服务器安装验证。

测试参数集中在项目的 `config.json` 中。直接运行 `python app.py` 会自动读取该文件，默认设置保持不变。优先级：命令行参数 > 自定义配置 > 项目 config.json。

```powershell
# 使用独立实验配置（可只填写需要覆盖的字段）
python app.py --config experiment.json
# 临时覆盖采样帧数
python app.py --config experiment.json --max-frames 24
# 仅检查配置对应的数据筛选结果
python app.py --config experiment.json --check-data
```

例如，保存以下内容为 `experiment.json`，即可测试所有标注时长不超过 60 秒的 Recognition 题目：

```json
{
  "max_duration": 60,
  "task_domain": ["Recognition"],
  "limit": 0
}
```

配置字段与命令行参数对应，名称使用下划线：`video_id`、`task_type`、`max_frames` 等。筛选列表填数组，`null` 表示不启用该筛选；`output: null` 表示自动生成结果文件。配置中的相对路径以该配置文件所在目录为基准，命令行中的相对路径以当前工作目录为基准。

`fps` 是目标采样频率，仍受 `max_frames` 上限限制；`use_audio_in_video` 控制输入音轨，`do_sample` 控制是否随机生成。`gpu_memory`、`cpu_memory` 是权重放置预算；`attn_implementation` 默认 `sdpa`。可用 `--no-short-only`、`--no-use-audio-in-video`、`--no-do-sample` 覆盖配置中的布尔开关。结果 JSON 的 `config.effective_parameters` 保存最终生效参数。未知字段或类型错误会在加载模型前报错。

使用已有的 `omnizip` 环境，默认运行标注顺序中的第一道题：

```powershell
conda activate omnizip
python app.py
```

也可不激活环境直接运行：

```powershell
conda run --no-capture-output -n omnizip python app.py --limit 3
```

IDE 的解释器应选择 `C:\Users\wyz\.conda\envs\omnizip\python.exe`。base 环境未安装 torch，但支持数据检查和单元测试。

## 数据格式

```text
D:\Datasets\WorldSense\
  worldsense_qa.json
  videos\
    dvOkwKAs.mp4
    UYkFSXsh.mp4
    ...
```

程序按 JSON 中的视频顺序、各视频的 task0/task1 等数字顺序展开题目。每道题输入完整视频的抽样画面、完整音轨、question 和 candidates；answer 只用于判分，video_caption 不输入模型。固定使用贪心解码，要求只输出选项字母。无法明确解析的回答计为错误，不猜测选项。

## 常用命令

```powershell
# 检查所有标注及对应视频文件是否存在（不加载模型）
python app.py --check-data

# 验证前三道题
python app.py --limit 3

# 只检查/测试 30 秒及以下的视频；--limit 0 表示该子集中的全部题目
python app.py --short-only --check-data
python app.py --short-only --limit 0 --output results/worldsense_short_30s.json

# 也可以指定任意时长上限（秒）
python app.py --max-duration 45 --limit 10

# 指定视频，运行该视频的所有题目
python app.py --video-id UYkFSXsh --limit 0

# 指定视频，只运行其中的若干 task
python app.py --video-id aVOJIFgy --tasks task0 task2 --output results/aVOJIFgy_task0_task2.json

# 按 Task Domain 测试
python app.py --task-domain Recognition --limit 10 --output results/recognition_sample.json
python app.py --task-domain Recognition Reasoning --limit 0 --output results/recognition_reasoning.json

# 按 Task Type 测试（包含空格的名称需要加引号）
python app.py --task-type "Audio Recognition" --limit 10
python app.py --task-type "Audio Recognition" "Spatial Relation" --limit 0

# Task Domain 和 Task Type 可同时指定，结果取交集
python app.py --task-domain Recognition --task-type "Audio Recognition" --limit 0

# 从第 11 道题开始运行 5 道题，并指定新结果文件
python app.py --offset 10 --limit 5 --output results/batch_11_15.json

# 全量运行 3172 道题（耗时很长，本次未执行）
python app.py --limit 0

# 更换数据集或模型目录
python app.py --dataset D:\Datasets\WorldSense --model D:\Models\Qwen2.5-Omni-3B

# 不需要模型依赖的单元测试
python -m unittest -v test_app
```

`--limit` 按题计数，默认 1，0 表示所有剩余题目；`--offset` 从 0 开始，在视频筛选之后应用。不指定输出路径时，会在项目 `results/` 下创建带时间戳的 JSON。已有结果文件不会被覆盖，请换用新的文件名。程序每完成一题就保存一次，包含原始回答、预测选项、标准答案、正确性、耗时、运行参数和汇总。准确率分母包含已处理的失败题和无法解析的题，未处理的题不会纳入分母。错误和未解析答案会导致最终退出码非零；正常运行但答错不会导致非零退出码。

`--short-only` 使用 JSON 中的 `video_duration` 秒数筛选完整视频，因此不会把同一视频的题目拆开。按数据集标注值，该子集包含 39 个视频、78 道题；`--short-only --limit 0` 会完整评测这个子集。标注时长以整秒记录，若需要按解码后的精确时长筛选，应先使用前面的全量时长统计脚本核对边界样本。`--check-data` 只校验数据和筛选结果，不加载模型。

`--tasks` 必须和 `--video-id` 一起使用，接受一个或多个 task 编号，例如 `task0 task2`。程序会先按视频编号筛选，再按 task 编号筛选，最后才应用 `--offset` 和 `--limit`；因此不会混入其他视频的同名 task。

每次结果的 `summary` 都包含 `accuracy_by_task_domain` 和 `accuracy_by_task_type`。每个类别包括 `correct`、`total` 和 `accuracy`；统计分母包含已处理的错误和无法解析的回答。`--task-domain` 和 `--task-type` 先筛选题目，再执行 `--offset`、`--limit`。

## 4-bit 量化测试

`config.json` 的 `load_in_4bit` 默认 `false`，改为 `true` 可开启 bitsandbytes NF4 权重量化，启用双重量化，计算精度在 GPU 支持时为 BF16，否则为 FP16。也可以临时覆盖：

```powershell
python app.py --load-in-4bit --limit 1 --output results/quant4_sample.json
python app.py --no-load-in-4bit --limit 1 --output results/full_precision_sample.json
```

两种模式均使用项目本地模型实现。量化模式需要 CUDA 和兼容的 bitsandbytes；不满足条件会报错，不会静默退回非量化。量化由 Transformers 对支持的层执行，未量化层、激活和 KV cache 不保证为 4-bit。仍采用 `device_map=auto` 及配置的内存预算；若预算不足导致量化层被分配到 CPU/磁盘，加载可能报错，需要调整预算，本实现不自动开启混合精度 CPU 卸载。视频采样参数不会因量化自动变化。

结果 `config.quantization` 保存量化配置，`config.is_loaded_in_4bit` 记录实际加载状态。比较效果时应固定视频、题目和采样设置；输出文件请使用新文件名。

## 输入 token 统计

每次模型输入会打印 token 数，并在逐题结果的 `input_tokens` 中保存 `video_tokens`、`audio_tokens`、`query_tokens` 和 `total_tokens`。统计依据是处理器展开多模态占位符后的实际 `input_ids`，排除 padding 和生成的答案。视频/音频 token 指供语言模型接收相应特征的位置数，不是原始像素、音频采样点或编码器中间序列长度。

`query_tokens` 定义为剩余的非视频、非音频输入 token，包含系统提示、问题、选项、作答指令、对话模板和模态边界标记，不仅是问题句子的长度。四项满足 `total_tokens = video_tokens + audio_tokens + query_tokens`。

总结的 `summary.input_tokens.averages` 给出各项平均值。按每次视频-task 输入平均，同一视频的不同 task 分别计入；`counted_inputs` 为实际测得 token 数的输入数。预处理失败时逐题值为 null，不作为 0 纳入平均，并计入 `uncounted_inputs`；若预处理成功但生成失败，则保留并统计已测得的输入。未测得任何输入时平均值为 null。关闭音频时音频 token 数为 0。

## 显存与评测范围

延续本机 8 GB 显卡的设置：加载时禁用语音生成、使用 SDPA、权重最多占用 5 GiB 显存，其余放到 CPU；默认最多均匀采样 16 帧，每帧最多 100352 像素。实际采样帧率会传给处理器，以保持音画时间对齐。音轨不主动裁剪，无音轨或损坏的视频会记录为错误。较长视频仍可能超出显存，错误会被保存并继续下一题；需要时可按视频进行单独诊断。

这些设置降低了视觉细节，也会因 CPU 分担计算而变慢。少量样本结果仅用于验证本地流程，不能视作全数据集准确率或官方榜单成绩。可通过 `--max-frames`、`--max-pixels`、`--max-new-tokens` 调整输入规模与回答长度。

当前已验证环境沿用 PyTorch 2.6.0+cu124 和 Transformers 4.52.3，以及现有环境的 qwen_omni_utils。

## 本地模型修改

`app.py` 从项目内的 `modeling_qwen2_5_omni.py` 导入 `Qwen2_5OmniForConditionalGeneration`。该文件包含可直接修改的模型实现，包括 Thinker、视觉编码器、音频编码器、注意力及生成逻辑；配置类、公共工具和 `Qwen2_5OmniProcessor` 仍来自 Transformers。

本地实现以当前环境的 Transformers 4.52.3 模型文件（含环境中已有修复）为基础，将包内相对导入改为绝对导入；没有修改环境安装目录。原来的较新版本文件依赖当前环境缺少的内部接口，已备份为 `modeling_qwen2_5_omni.py.original.bak`。许可证保留在源码中。后续修改本地模型文件后，重新启动测试进程即可生效；升级 Transformers 时需重新检查接口兼容性。
