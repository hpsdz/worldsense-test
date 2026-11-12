# Qwen2.5-Omni 本地 WorldSense 评测

统一策略支持 baseline、OmniZip、OmniSelect、random、OmniPack 和 OmniScope。
OmniScope 使用 `python app.py --config configs/omniscope.json`，额外权重与参数见 [接入说明](docs/methods/omniscope.md)。

默认从 `D:\Datasets\WorldSense` 读取视频和选择题，模型路径为 `D:\Models\Qwen2.5-Omni-3B`。

## 辅助模型资源

将 `resources.example.json` 复制为 `resources.local.json`，集中设置 AudioCLIP、CLIP 和 NLTK 路径。
方法配置无需包含本机绝对路径。资源迁移、环境变量与 Git 历史注意事项见 [资源说明](docs/RESOURCES.md)。

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

使用已有的 `omnizip` 环境，按 config.json 的筛选和题数运行：

```powershell
conda activate omnizip
python app.py
```

也可不激活环境直接运行：

```powershell
conda run --no-capture-output -n omnizip python app.py --limit 3
```

IDE 的解释器应选择 `C:\Users\wyz\.conda\envs\omnizip\python.exe`。请使用安装了 requirement.txt 的解释器运行测试；编辑器提示无法解析 nltk 时也应先核对解释器。

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

# 回归测试（使用已安装项目依赖的环境）
python -m unittest discover -s tests -v
```

`--limit` 按题计数，默认值由 config.json 决定，0 表示所有剩余题目；`--offset` 从 0 开始，在视频筛选之后应用。不指定输出路径时，会在项目 `results/` 下创建带时间戳的 JSON。已有结果文件不会被覆盖，请换用新的文件名。程序每完成一题就保存一次，包含原始回答、预测选项、标准答案、正确性、耗时、运行参数和汇总。准确率分母包含已处理的失败题和无法解析的题，未处理的题不会纳入分母。错误和未解析答案会导致最终退出码非零；正常运行但答错不会导致非零退出码。

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

## 共享模型和目录

app.py 直接导入 models/qwen2_5_omni/modeling_qwen2_5_omni.py；根目录不再保留同名转发文件。
后续模型修改集中在该共享实现。配置类和 Processor 仍来自 Transformers 4.52.3，不建议单独升级。

- app.py：统一评测入口；config.json 和 configs/：公共及方法参数。
- models/：共享模型；compression/：策略适配和核心算法。
- tests/：必要的回归测试，不是数据集运行入口。
- docs/methods/：方法专用说明；docs/history/：历史迁移记录。
- docs/upstream/：上游说明和许可证；docs/legacy/：旧源码与历史资料。
- vendor/：第三方依赖；assets/：运行资源；results/：实验结果。

文档导航见 [docs/README.md](docs/README.md)。第三方目录中的 README 和许可证保留原位置，以满足包构建和来源追溯。

## 统一压缩评测

baseline、OmniZip、OmniSelect、random、OmniPack、OmniScope 共用
`models/qwen2_5_omni/modeling_qwen2_5_omni.py`、本地模型权重、vendor
预处理库、问题模板、生成参数与评分器。app.py 直接导入共享模型。
统一评测不再依赖 OmniZip-main/OmniSelect 旧目录。

## 运行

在 qwen 根目录、现有 omnizip 环境运行：

```powershell
python app.py --config configs/baseline.json
python app.py --config configs/omnizip.json
python app.py --config configs/omniselect.json
python app.py --config configs/random.json
python app.py --config configs/omnipack.json
python app.py --config configs/omniscope.json
```

这些配置叠加到根目录 config.json，保留其现有帧数、时长筛选、题数等设置。
压缩参数是整体替换的 JSON 对象，遗漏参数使用策略默认值。也可以直接指定：

```powershell
python app.py --compression-method omnizip --compression-params '{"rho_audio":0.4,"rho_video":0.6}'
python app.py --config configs/omniselect.json --video-id FnOIAada --tasks task0 task1 --max-frames 16 --max-new-tokens 4
```

OmniSelect 需要已下载的 AudioCLIP 权重、词表和 NLTK 资源，详见
`docs/legacy/OmniSelect.md`（历史记录）；权重通过 `resources.local.json` 配置，词表位于 `vendor/audioclip/assets/`，
NLTK 资源位于 `assets/nltk_data/`。默认 AudioCLIP 在 CPU 上逐题加载计算，避免与
Qwen 争用 8 GB GPU 显存；配置 compression_params.audioclip_device 可改变设备。
这种部署不是论文性能配置。辅助模型的加载和计算均计入策略准备耗时及总耗时。

### 随机删除对照

`configs/random.json` 选择 `random` 策略。`prune_ratio_v` 和 `prune_ratio_a`
分别表示视频和音频删除比例，范围 [0,1)，默认 0.6 / 0.4；设为 0 可保留该模态。
每个模态全序列均匀无放回抽取 `floor(token数 × 删除比例)` 个位置删除，
不分时间组，不合并 embedding，不改变剩余 token 顺序，保留所有文本与特殊 token。
因向下取整，实际删除率可能略低于配置值；总序列删除率还取决于不删除的文本比例。

`seed` 默认 0，范围为 [0,2**63) 的整数。每次调用重建独立 CPU 随机生成器，
视频和音频使用不同随机流，不修改 PyTorch 全局随机状态，不依赖题目处理顺序。
相同种子、相同模态 token 数将得到相同模态内删除位置（即使问题不同）；
要评估随机波动，使用多个 seed 分别运行。它不计算音频重要性，也不运行 AudioCLIP。
其余缓存处理、输入限制、耗时和 token 统计共用现有统一流程。

## 结构与扩展

OmniScope 的额外 CLIP 权重、参数及与上游的适配差异见 [docs/methods/omniscope.md](docs/methods/omniscope.md)。

- app.py：数据筛选、统一预处理、策略准备、推理和结果保存。
- compression/strategies.py：注册表、策略接口、基础策略适配器。
- compression/algorithms/：从各上游实现提取的压缩算法，不含 Qwen 网络。
- compression/relevance.py：OmniSelect AudioCLIP 查询相关性与实际 token 分组对齐。
- compression/runtime.py：按需提取音频重要性、应用压缩结果、缓存位置适配。
- models/qwen2_5_omni/modeling_qwen2_5_omni.py：共享模型，调用压缩接口。
- compression/omnipack.py：OmniPack 两阶段适配、编码器注意力统计与缓存对齐。
- compression/algorithms/omnipack/：上游的两阶段选择与合并算法。

新方法继承 Strategy，用 @register_strategy('method_name') 注册，并在策略模块
导入其实现。定义 defaults、prepare(**kwargs)、compress(...)：

1. prepare 接收同一份采样帧、16 kHz 音频、问题和 Processor 输出；可返回辅助信息。
2. 如需音频重要性，设置 needs_audio_importance=True；baseline 不计算此项。
3. compress 返回 CompressionResult：完整长度的 embedding（允许合并）与布尔保留 mask。
4. 公共运行时负责筛选 embedding/mask/位置并统计 token；要求所有文本与特殊 token 保留。

Strategy.compress 合同是 **语言模型输入前的压缩**。OmniPack 另外使用共享 TextModel
的层后调用完成第二阶段，并同步裁剪所有已计算层的 KV 缓存。其他层内方法应参照此适配，
不要把第二阶段移到输入处。KV 驱逐、多视频、多样本 batch 需要额外接口。
当前压缩适配器限推理、单视频、无 padding、音视频
同时存在、SDPA、DynamicCache。baseline 保留原有输入能力。4-bit 开关仍可用，
但本轮没有验证压缩与 4-bit 的组合。

## 位置和算法说明

预填充保留原多模态 RoPE 语义位置；KV 的物理存储位置压缩为连续位置。
生成阶段保留原始语义时间尺度，裁剪原提示词 mask，并修正物理 cache_position。
每题重置压缩 mask、重要性和统计，避免上一题污染下一题。

音频重要性采用原项目的全局 Q/K 分块 softmax 汇总方式；它并非修改编码器
实际 attention mask。通过临时 hook 提取最后一层输入，再执行共享编码器。
当前每两项重要性平均池化；与实际音频输出长度不匹配时明确报错，不静默截断。

OmniSelect 使用此前本地 demo 的时间组修复，包括稀疏帧与音视频交错产生的碎片组。
OmniZip 保留其原算法的预算分配，因此相同名义压缩率不等于相同实际 token 数。
缓存修正可能改变相对旧 demo 的生成结果；本实现不是逐字复刻原脚本输出。

## 统计与验证

每题保留原 input_tokens，并新增 compression_tokens（各模态 before/after）、
strategy_prepare_seconds、generation_seconds、generated_tokens、sampled_frames、
sampled_fps、peak_cuda_allocated_bytes；summary.compression 给出已测题目的平均数。
失败题保留原错误机制，不以 baseline 数据替代失败压缩结果。

elapsed_seconds 包括预处理、策略准备、生成、解码；generation_seconds 包括 Qwen
编码、重要性提取、压缩、预填充和解码，不含外部 AudioCLIP。
显存是当前 CUDA 设备 PyTorch allocated 峰值，不是多卡总量或整卡占用。

### 全局和平均耗时

新运行的结果在 `summary.timing` 中保存：

- `global.model_load_seconds`：Qwen 模型及 Processor 加载与设备同步。
- `global.evaluation_wall_seconds`：初次结果保存开始，到最后一题保存、清理及设备同步结束。
- `global.run_wall_seconds`：进入 main 到上述结束点，包含参数解析、数据验证、依赖导入、模型加载及评测。
- `global.finalized`：逐题检查点为 false，正常结束（包括已捕获的单题失败）为 true。
- `all_processed` / `completed`：分别汇总所有已处理题目，以及无推理异常的题目
  （包括 invalid_answer）；各字段提供 count、missing_count、total_seconds、mean_seconds、median_seconds。

汇总字段包括 elapsed_seconds（每题端到端）、strategy_prepare_seconds（策略准备）和
generation_seconds（Qwen 编码、压缩、预填充与生成）。缺失阶段不按零计入平均值；空集合为 null。
单题耗时保留浮点精度，不再先四舍五入至两位小数。策略准备已完成但后续生成失败时，保留准备耗时。

全局耗时不包括 Python 启动及 main 之前的模块导入、最终汇总文件的写入/打印和进程退出。
这是明确截止点的墙钟时间，不是逐题耗时之和。中断时已有文件仅包含最近检查点的时间；旧结果
没有全局字段，不能从逐题耗时反推出完整全局时间。模型加载失败、check-data 模式不生成评测报告。

已用 FnOIAada/task0、task1（16 帧、最多生成 4 token）分别完成三种方法的真实推理，
每种连续两题，无异常和无效回答。平均 token：

| 策略 | 视频 | 音频 | Query | 总计 |
|---|---:|---:|---:|---:|
| baseline | 1152 | 600 | 126.5 | 1878.5 |
| OmniZip | 709 | 390 | 126.5 | 1225.5 |
| OmniSelect | 459 | 360 | 126.5 | 945.5 |

结果位于 results/unified_*_validation.json。两题仅为接入验证，不能据此比较准确率。
运行 `python -m unittest discover -s tests -v` 检查配置、原评测功能、
策略边界和物理缓存/语义位置对齐。

另外使用相同随机权重的小型 Thinker、显式位置编号对比改造前后 baseline，
得到逐元素完全相同的 logits；该数值检查是文本路径检查，不是全模型多模态
全数据集等价性证明。原始音视频编码器和权重结构未改写。
