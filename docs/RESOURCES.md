# 资源配置与 Git 管理

将根目录 resources.example.json 复制为 resources.local.json。每台机器只修改这一份本地配置，
无需分别修改方法 JSON 或 Python 源码。local 文件被 Git 忽略，不会覆盖其他人的部署路径。

```json
{
  "audioclip": "/data/models/AudioCLIP/AudioCLIP-Full-Training.pt",
  "clip_model": "/data/models/clip-vit-large-patch14-336",
  "nltk_data": "assets/nltk_data"
}
```

优先级：方法 compression_params 中显式路径 > QWEN_AUDIOCLIP / QWEN_CLIP_MODEL / QWEN_NLTK_DATA
环境变量 > resources.local.json > resources/ 默认目录（NLTK 默认 assets/nltk_data）。
本地资源文件和环境变量中的相对路径以项目根目录为基准，支持 ~ 和环境变量展开。
为兼容已有实验，方法 JSON 中的显式路径仍相对该 JSON；模型、数据、输出路径也保持原规则。
主 Qwen 模型和数据集仍在 config.json 配置。实验结果保存解析后的辅助资源路径。

```bash
python -m compression.resources --method omniselect
python -m compression.resources --method omniscope
```

以上检查文件布局，不加载大模型，不等于权重内容完整性校验。
AudioCLIP 权重来自 https://github.com/AndreyGuzhov/AudioCLIP ，CLIP 来自
https://huggingface.co/openai/clip-vit-large-patch14-336 。下载后填入本地路径。

本机 AudioCLIP 权重已从 vendor 移到 D:/Models/AudioCLIP/AudioCLIP-Full-Training.pt，
SHA256 为 2441d35b353352c8b1bbfb8f7c687f46314c3d2909e940eaf763b8c17f632c44。
vendor/audioclip 保留小型词表、推理代码和许可证，避免重新下载词表才能导入模型。
权重格式、resources/、本地配置及缓存已加入 .gitignore；原已跟踪权重已从当前 Git 索引移除。

注意：历史提交仍含旧权重，普通新提交不会缩小 Git 历史，也可能仍被 GitHub 的大文件限制拒绝。
此次不改写历史；首次公开发布可从当前清理后的文件建立新仓库（排除 .git 和所有忽略项），
或者另行备份、协调后使用 git-filter-repo 清理历史再推送。不要把旧 .git 一起复制到新仓库。
