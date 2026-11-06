# 0032 — 控制组验证结果与六模型对比

日期：2026-11-06（America/New_York）。

用户完成 evaluate_torchvision_controls.py。选定本次回传的两个报告：

- control_resnet50_1nn_validation_20261106_160802_791716.json：77/93，Accuracy
  0.8279569892473119，Macro Precision 0.8404940046193916，Macro Recall
  0.8293359010270774，Macro-F1 0.8326187084742344，Weighted-F1 0.8299795865980515。
- control_convnext_tiny_1nn_validation_20261106_160802_886467.json：77/93，Accuracy
  0.8279569892473119，Macro Precision 0.8336192810457517，Macro Recall
  0.8321311858076564，Macro-F1 0.8317731526736903，Weighted-F1 0.8279391790491339。

两者参数更新 0、未训练分类器、使用带标签训练参考、不是零样本，测试集未评估。
报告使用共享评估器的旧 Asia/Shanghai 时间偏移；本实验记录按客户端 America/New_York 日期记载，
不修改已生成的原始报告。

独立核验：缓存与 CSV 哈希、模型/权重/版本、434 个参考和 93 个查询编号、类别顺序、
逐图真实标签/预测/最近邻/余弦相似度、五项整体指标、逐类指标及混淆矩阵均一致。
项目中还存在 2026-11-05 运行生成的同模型报告及 CSV；两次运行的全部预测、最近邻、
指标和混淆矩阵完全一致，证明固定缓存评估是确定的。重复运行不当作独立实验或重复样本。

逐类结果：ResNet50 的 F1 为 geometric 90.32%、animal 74.29%、antique object 90.32%、
plant 75.86%、Chinese character 92.31%、combined 76.47%。ConvNeXt-Tiny 对应为
90.32%、70.97%、93.75%、75.00%、96.30%、72.73%。

六模型验证 Macro-F1 排序：DINOv3 83.45%、ResNet50 控制组 83.26%、ConvNeXt-Tiny
控制组 83.18%、SigLIP 2 82.94%、EUPE 82.75%、Qwen 2B 78.77%。除 Qwen 外其余五款
均为 77/93。DINOv3 仅比 ResNet50 控制组高 0.1850 个百分点，比 ConvNeXt-Tiny 控制组
高 0.2696 个百分点；当前固定验证集不能支持“新型预训练模型稳定优于普通 ImageNet 监督预训练”的结论。

逐图比较也显示错误互补：DINOv3 与 ResNet50 共同正确 67、各自独有正确 10、共同错误 6，
预测类别不同 23 张；DINOv3 与 ConvNeXt-Tiny 共同正确 69、各自独有正确 8、共同错误 8，
预测类别不同 20 张。总分接近不表示错误模式相同。

MODEL_COMPARISON.md、README 和两个控制组配置已更新。DINOv3 按预定 Macro-F1 仍暂列第一，
但优势不足 0.3 个百分点，最终 baseline 尚未选定，也没有开始微调或测试集评估。
原图、权重和特征缓存保留本地；GitHub 仅上传配置、代码、小型日志和逐图预测表。
