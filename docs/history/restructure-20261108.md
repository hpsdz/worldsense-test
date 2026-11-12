> 历史快照，路径和清理状态以当前根目录 README 为准。

# 统一项目目录与 OmniPack 接入（2026-11-08）

参考相邻 OmniPack 项目把共享模型、策略、算法、配置和第三方资源分离，保留原测试入口和结果格式。

```text
app.py                       WorldSense 统一评测入口
config.json / configs/       通用参数 / 各方法参数
models/qwen2_5_omni/          唯一共享模型实现
modeling_qwen2_5_omni.py      旧导入路径兼容入口
compression/                 策略注册、模型适配、AudioCLIP 相关性
compression/algorithms/      OmniZip、OmniSelect、OmniPack 核心算法
vendor/audioclip/            OmniSelect 所需辅助模型代码和权重
vendor/qwen-omni-utils/      所有策略统一预处理
assets/                     示例视频、NLTK 资源
docs/upstream/              原作者说明、许可证
docs/legacy/                历史 demo 说明、旧源码压缩包、Git 历史
results/                    所有测试记录（保留）
```

## OmniPack

```powershell
python app.py --config configs/omnipack.json --video-id FnOIAada --tasks task0 --limit 1
python -B -m unittest test_app test_compression test_omnipack
```

参数含义：

| 参数 | 含义 |
| --- | --- |
| video_ratio / audio_ratio | 最终目标保留比例（不是删除比例），默认 0.2 / 0.4 |
| encoder_ratio_scale | 第一阶段预算倍率，第一阶段保留 min(1, 最终比例 × 倍率)，默认 2 |
| grid_in_window | 视觉注意力重要性统计窗口的时间网格数，默认 2；一格通常是两帧 |
| sec_in_audio_window | 音频注意力统计窗口秒数，默认 2，对应下采样前 100 token |
| stage2_layer | 完成第几层后执行第二阶段，从 1 开始，默认 18；必须小于模型总层数 |

第一阶段使用上游的注意力引导选择、时空/时间冗余合并；第二阶段使用文本问题和跨模态相关性选择并合并。
第二阶段相对第一阶段的保留比例为 最终比例 / 第一阶段比例；各阶段四舍五入且模态至少保留一个，因此实际个数存在取整误差。
配置中的比例是本项目实验默认值，不声称是论文最佳设置。

上游编码器补丁只支持 FlashAttention2；本适配保留现有 SDPA 编码器，临时 hook 最后层的 Q/K，按上游公式计算窗口注意力统计。
视觉统计含旋转位置编码、空间合并及窗口逆排序，音频统计按两倍下采样对齐。窗口计算分块以降低临时显存。
核心 pre_llm_units.py / inner_llm_units.py 原样复制；没有复制上游全模型补丁、lmms-eval 或训练框架。
尚未做与上游 FlashAttention2 的逐数值等价实验，不能把此处单样本结果当作论文复现。

统计字段 `compression_tokens.*` 的 `before` 是原输入，`after_stage1` 是前 18 层的输入数，`after` 是后续层及最终缓存保留数。
平均值同样包含三阶段。不能把 `after` 当作所有层均使用的输入长度；前面层已完成的计算不会因第二次删除而消失。
查询 token 计数沿用原系统，包含文本和特殊标记；第二阶段上游的文本相关性也使用全部非音视频位置。

## 迁移与清理状态

已经迁出 AudioCLIP 权重、词表、NLTK 资源和示例视频；权重迁移前后 SHA256 一致。
旧源码（包括本地修改和 demo）归档为 `docs/legacy/legacy-sources.zip`，OmniSelect Git 历史为 `omniselect-history.bundle`。
历史 `OmniSelect/results/demo.json` 复制到 `results/legacy/`，修改过的 infer_worldsense.sh 单独保存在 docs/legacy。
上游说明和 OmniZip/AudioCLIP 许可证保留。没有修改相邻 OmniPack 仓库，也没有修改用户 config.json 参数。

计划移除的冗余项包括旧目录中的重复 Qwen 模型、lmms-eval、eval、训练入口、重复预处理库、下载碎片和缓存。
删除操作被自动审批策略拒绝（blocked by policy），因此 **OmniZip-main 和 OmniSelect 仍实际存在**，尚未释放其占用空间。
vendor/audioclip 中复制的 datasets、_trainer.py 和缓存也未删除，但推理入口已不导入 datasets。
无需这些旧目录即可运行新入口；旧 demo 的原始依赖关系只作为历史记录保存。

## 验证

27 项单元测试通过，覆盖原评测、现有策略、OmniPack 窗口统计、两阶段 token 数、阶段平均值、文本保护、KV 裁剪和连续解码。
现有 omnizip 环境、3B、SDPA、非量化、16 帧下 FnOIAada/task0 实际生成 B.，答案正确：
视频 1152 → 461 → 230；音频 600 → 480 → 240；文本 119 不变；总计 1871 → 1060 → 589。
生成耗时约 18.03 秒，仅为当前机器单次样本，不能推断总体准确率或速度。
未验证 OmniPack 与 4-bit 的组合或 7B 模型。

迁移后 OmniSelect 同一任务实际推理成功，视频 459、音频 360、文本 119，预测 A（错误）；记录见 results/omniselect_migration_validation_20261108.json。
OmniZip 同一任务实际推理成功，视频 709、音频 390、文本 119，预测 C（错误）；记录见 results/omnizip_migration_validation_20261108.json。
