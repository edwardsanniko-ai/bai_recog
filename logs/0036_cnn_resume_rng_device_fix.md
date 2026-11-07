# 0036 — CNN checkpoint 随机状态恢复设备修复

日期：2026-11-07（America/New_York）

用户从 seed 42 第 1 轮的 `last.pt` 恢复时，在真正进入第 2 轮前出现：

```text
TypeError: RNG state must be a torch.ByteTensor
```

原因是训练脚本使用 `torch.load(..., map_location=device)` 将 checkpoint 中的所有张量映射到 CUDA，其中也包括原本属于 CPU generator 和 DataLoader generator 的随机状态；`torch.set_rng_state()` 只接受 CPU ByteTensor。checkpoint 本身、模型参数、优化器状态和第 1 轮结果均未损坏。

修复只作用于 `restore_rng_state()`：CPU RNG、DataLoader RNG 以及保存的 CUDA RNG 状态在交给 PyTorch generator API 前显式恢复为 CPU `torch.uint8` 张量。模型和优化器仍按原路径映射到训练设备，不改变实验配置、数据划分、随机种子或已保存的第 1 轮 checkpoint。

使用原 checkpoint、CUDA map location 做了无训练恢复检查，结果：

```text
checkpoint_epoch: 1
loaded_map_location: cuda
cpu_rng_device: cpu
loader_rng_device: cpu
resume_rng_restore: PASS
```

检查过程没有运行 forward/backward、没有创建新预测、没有更新参数，也没有评估测试集。下一次恢复应从 epoch 2 开始。
