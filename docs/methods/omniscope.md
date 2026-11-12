# OmniScope 统一测试接入

```powershell
python app.py --config configs/omniscope.json --video-id FnOIAada --tasks task0 --limit 1
python -B -m unittest discover -s tests -v
```

继续叠加根目录 config.json 的数据筛选、帧数、分辨率、模型路径和生成参数。
服务器部署时修改根目录 resources.local.json 中 clip_model 为本地 CLIP ViT-L/14@336 权重目录。
本机 resources.local.json 指向 D:/Models/clip-vit-large-patch14-336；未配置时默认 resources/clip-vit-large-patch14-336。运行时只读取本地，不自动联网下载。

| 参数 | 含义 |
| --- | --- |
| visual_prune_ratio | 视频目标删除比例，默认 0.6，范围 [0,1) |
| audio_prune_ratio | 音频目标删除比例，默认 0.25，范围 [0,1) |
| clip_model | 官方 openai/clip-vit-large-patch14-336 的本地目录 |
| clip_device | 默认 cpu，减少与 Qwen 争抢显存；服务器可设 cuda |
| clip_batch_size | CLIP 图像编码批大小，默认 4 |

视觉：使用统一采样后的同一份视频帧，CLIP 图文余弦相似度每两帧平均，经过上游的非负三次方变换后分配时间网格预算。
各网格根据空间密度、相邻网格冗余选择 token；上游按宽高比取整预算且最少保留 2x2，因此实际删除率不一定等于目标值。
全零视觉分数使用均匀预算，修复上游除零问题；视觉删除比例为零时明确保留全部 token。

音频：用共享 Qwen 的原始问题词嵌入和音频编码特征计算门控文本相似度，每秒 25 token 中取 top-5 平均；
沿用上游平方变换、80 分位邻域增强、每秒至少 10 token 的预算分配，再做二分图相似性合并。
不足一秒的尾段保持原样，故音频实际删除率通常不超过约 60%。音频重要性不使用 OmniZip 的注意力分数。
CLIP 和音频语义评分沿用上游的“问题 + 选项”，不加入生成指令；生成仍使用统一选择题提示。

算法来自本地上游 d8fb72f，核心函数提取到 compression/algorithms/omniscope.py，
策略位于 compression/omniscope.py，不依赖 OmniScope/tools 的完整模型或自定义 Processor。
共享 runtime 负责最终 mask、token 统计、RoPE 和 KV 缓存。音频合并后的特征按上游均匀时间锚点放回原始序列，视频保留原空间位置。

与上游实验的差别：保留本项目统一采样、Processor、SDPA 和每题完整编码；没有复用上游跨题音视频 embedding 缓存。
CLIP 默认 CPU float32，每题加载并释放，计入 strategy_prepare_seconds；音频评分/合并在模型内执行，计入 generation_seconds。
原 generation 显存峰值统计在 prepare 后重置，不包含 CLIP 准备阶段峰值；elapsed_seconds 和全局时间包含准备开销。
这些选择用于统一测试，不声称与论文的准确率或速度等价。当前仍限 batch=1、单视频、音视频齐全、SDPA、DynamicCache。
未验证该策略与 4-bit 或 7B 的组合。

## 验证记录（2026-11-10）

现有 omnizip 环境、Qwen2.5-Omni-3B、SDPA、非量化、16 帧下测试 FnOIAada 的 task0/task1：

| 任务 | 视频 token | 音频 token | 文本 token | 预测/结果 |
| --- | --- | --- | --- | --- |
| task0 | 1152 → 469 | 600 → 450 | 119 不变 | B，正确 |
| task1 | 1152 → 483 | 600 → 450 | 134 不变 | A，错误 |

两题均正常完成，记录位于 results/omniscope_validation_final_20261110.json。只验证功能，不代表整体准确率。
30 项单元测试通过，覆盖已有策略以及新方法的预算、音频合并、尾段保护、文字保护和零剪枝。
官方 CLIP 权重已下载到默认路径，1711974081 字节，SHA256：
`c6032c2e0caae3dc2d4fba35535fa6307dbb49df59c7e182b1bc4b3329b81801`。
