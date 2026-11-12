# 本地推理适配

AudioCLIP 保留上游 LICENSE.md。项目使用 vendor.audioclip 包路径导入，
内部 model、utils_a、ignite_trainer 引用统一限定包名，不再修改 sys.path
或依赖运行目录。ignite_trainer 的训练入口保持惰性导入；推理不导入 datasets。
权重和词表仍放在 assets/，模型结构及权重格式不变。
