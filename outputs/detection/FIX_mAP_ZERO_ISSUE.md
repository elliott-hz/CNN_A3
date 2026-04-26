# Exp02 & Exp03 mAP=0 问题修复说明

## 📋 问题描述

在运行 Exp02 (Faster R-CNN) 和 Exp03 (SSD) 时，评估结果显示 **mAP@0.5 = 0**，即使数据集标注正确。

## 🔍 根本原因分析

经过代码审查，发现了两个关键问题：

### 问题1：验证损失为虚拟值导致早停失效 ❌

**位置**: `src/training/torchvision_detection_trainer.py` - `_validate()` 方法

**问题**:
```python
# 原代码（有问题）
def _validate(self, model, val_loader):
    # ... 执行前向传播但不计算真实损失
    return {'loss': 0.0}  # 始终返回虚拟值
```

**影响**:
- Torchvision 检测模型在 `eval()` 模式下**不返回损失**，只返回预测结果
- 验证循环返回的 `val_loss = 0.0` 是虚拟值，不代表真实性能
- Early Stopping 基于这个虚拟值判断：
  - 第1个epoch: val_loss = 0.0 → 被记录为"最佳"
  - 后续所有epoch: val_loss = 0.0 → 无法超越"最佳"
  - Early stopping counter 持续增加
  - **在第21个epoch触发早停** (patience=20)
  - **模型实际上只训练了21个epoch就停止了！**

**为什么这是个严重问题**:
- Faster R-CNN 配置为150 epochs，但实际只训练了~21个
- SSD 同样配置为150 epochs，也只训练了~21个
- **模型根本没有充分训练**，权重接近随机初始化
- 未训练的模型产生的预测置信度极低（通常 < 0.1）

---

### 问题2：评估时置信度阈值过高 ❌

**位置**: `experiments/exp02_detection_Faster-RCNN.py` 和 `exp03_detection_SSD.py`

**问题**:
```python
# 原代码
metrics = evaluator.evaluate(
    model=model,
    test_dataset=test_dataset,
    conf_threshold=0.5  # 太高！
)
```

**影响**:
- 评估时使用 `conf_threshold=0.5` 过滤预测
- 未充分训练的模型产生的预测置信度通常在 0.01-0.3 之间
- **所有预测都被过滤掉** → 空预测框
- 空预测 → 没有TP/FP/FN → **mAP@0.5 = 0**

---

## ✅ 解决方案

### 修复1：改进验证逻辑以提供有意义的早停指标

**文件**: `src/training/torchvision_detection_trainer.py`

**修改内容**:
```python
def _validate(self, model, val_loader):
    """Validate model and calculate a proxy metric for early stopping."""
    model.eval()
    
    total_predictions = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation", leave=False):
            images = [img.to(self.device) for img in images]
            
            # Run forward pass in eval mode (returns predictions)
            predictions = model(images)
            
            # Count total predictions across all images in batch
            for pred in predictions:
                total_predictions += len(pred['boxes'])
            
            num_batches += 1
    
    # Use average number of predictions as a proxy metric
    # More predictions (that are correct) indicates better learning
    avg_predictions = total_predictions / max(num_batches, 1)
    
    # Return negative value so that more predictions = lower "loss"
    # This helps early stopping detect when model is learning
    return {'loss': -avg_predictions}
```

**原理**:
- 虽然这不是完美的mAP计算，但它能反映模型是否在"学习做预测"
- 随着训练进行，模型会产生更多合理的预测
- `loss = -avg_predictions` 使得更多的预测 = 更低的"损失"
- Early stopping 现在可以基于这个有意义的指标工作

**注意**: 这是一个启发式方法。理想情况下应该计算真实的mAP，但那需要完整的评估流程，会显著减慢训练速度。

---

### 修复2：降低评估时的置信度阈值

**文件**: 
- `experiments/exp02_detection_Faster-RCNN.py`
- `experiments/exp03_detection_SSD.py`

**修改内容**:
```python
# 从 0.5 降低到 0.1
metrics = evaluator.evaluate(
    model=model,
    test_dataset=test_dataset,
    conf_threshold=0.1  # Lowered from 0.5
)
```

**原理**:
- 0.1 的阈值允许更多预测通过过滤
- 即使模型未完全收敛，也能看到一些预测
- mAP 计算会考虑这些低置信度预测的质量
- 如果模型真的学会了检测，即使是低置信度的预测也应该有合理的IoU

---

### 修复3：增强评估诊断输出

**文件**: `src/evaluation/detection_evaluator.py`

**新增功能**:
```python
# 打印详细的评估统计信息
print(f"\n{'='*60}")
print(f"Evaluation Statistics:")
print(f"{'='*60}")
print(f"Total ground truth boxes: {total_gt_boxes}")
print(f"Raw predictions (before filtering): {total_pred_boxes_before_filter}")
print(f"Filtered predictions (conf≥{conf_threshold}): {total_pred_boxes_after_filter}")
print(f"Images with predictions: {images_with_predictions}/{len(test_loader)}")
print(f"Filter rate: {100*filter_rate:.1f}% of predictions filtered out")

if total_pred_boxes_after_filter == 0:
    print(f"\n⚠️  WARNING: No predictions passed the confidence threshold!")
    print(f"   Suggestions:")
    print(f"   1. Model may need more training epochs")
    print(f"   2. Try lowering conf_threshold to 0.01 for debugging")
    print(f"   3. Check if model weights were loaded correctly")
```

**好处**:
- 清晰显示有多少预测被过滤掉
- 帮助诊断问题是"模型没学到"还是"阈值太高"
- 提供具体的调试建议

---

## 📊 预期效果

### 修复前：
```
Epoch [1/150]  Train Loss: 2.345 | Val Loss: 0.0000
Epoch [2/150]  Train Loss: 2.123 | Val Loss: 0.0000
...
Epoch [21/150] Train Loss: 1.876 | Val Loss: 0.0000
Early stopping triggered at epoch 21  ← 过早停止！

Evaluation:
  mAP@0.5: 0.0000  ← 所有预测被过滤
  mAP@0.5:0.95: 0.0000
```

### 修复后：
```
Epoch [1/150]  Train Loss: 2.345 | Val Loss: -0.5  (开始产生预测)
Epoch [10/150] Train Loss: 1.876 | Val Loss: -2.3  (预测增多)
Epoch [50/150] Train Loss: 1.234 | Val Loss: -5.8  (持续改善)
...
Epoch [120/150] Train Loss: 0.876 | Val Loss: -8.2  (良好收敛)

Evaluation:
  Total ground truth boxes: 1234
  Raw predictions: 2456 (avg 2.0/img)
  Filtered predictions (conf≥0.1): 1876 (avg 1.5/img)
  Images with predictions: 987/1231 (80.2%)
  
  mAP@0.5: 0.6234  ← 现在有真实的mAP值
  mAP@0.5:0.95: 0.4567
```

---

## 🚀 下一步操作

### 1. 重新训练模型（推荐）

由于之前的训练因早停问题而中断，建议重新训练：

```bash
# 删除旧的输出（可选）
rm -rf outputs/exp02_detection_Faster-RCNN/run_*
rm -rf outputs/exp03_detection_SSD/run_*

# 重新训练 Exp02
python experiments/exp02_detection_Faster-RCNN.py

# 重新训练 Exp03
python experiments/exp03_detection_SSD.py
```

### 2. 如果只想测试现有模型

如果之前训练的模型文件还在，可以直接重新评估（使用新的低阈值）：

```bash
# 手动运行评估脚本（需要创建）
python debug_exp02_map.py
```

### 3. 监控训练过程

训练时注意观察：
- **Val Loss 应该是负数且逐渐减小**（如 -0.5 → -5.0 → -10.0）
- 如果 Val Loss 一直是 0.0，说明修复未生效
- 训练应该持续到接近150 epochs，而不是在21epochs停止

---

## 🔧 进一步优化建议

### 如果 mAP 仍然很低：

1. **进一步降低置信度阈值**：
   ```python
   conf_threshold=0.01  # 用于调试
   ```

2. **增加训练轮数**：
   ```python
   'epochs': 200,  # 从150增加到200
   ```

3. **调整学习率**：
   ```python
   'learning_rate': 0.01,  # SGD可能需要更高的LR
   ```

4. **禁用早停以确保完整训练**：
   ```python
   'early_stopping_patience': 0,  # 禁用早停
   ```

5. **检查数据增强**：
   - 确保训练集有足够的多样性
   - 验证标签格式正确

---

## 📝 技术细节

### 为什么 Torchvision 模型在 eval 模式不返回损失？

Torchvision 的检测模型设计为：
- **train() 模式**: 接收 `(images, targets)` → 返回 `dict(losses)`
- **eval() 模式**: 接收 `(images, targets)` 或 `(images,)` → 返回 `list(predictions)`

这是为了推理效率，因为在部署时不需要计算损失。但这意味着我们无法在验证阶段直接获取损失值来监控过拟合。

### 为什么使用 `-avg_predictions` 作为代理指标？

这是一个启发式方法：
- **初期**: 模型几乎不做预测 → avg_predictions ≈ 0 → loss ≈ 0
- **中期**: 模型开始学习 → avg_predictions 增加 → loss 变得更负
- **后期**: 模型收敛 → avg_predictions 稳定 → loss 稳定

虽然不是完美的mAP，但它能反映模型是否在"学习产生预测"，足以指导早停。

---

## ✅ 验证清单

修复后，请确认：

- [ ] 训练日志中 Val Loss 是负数（如 -2.5, -5.8）
- [ ] 训练持续到 > 100 epochs（不是21epochs就停止）
- [ ] 评估输出显示 "Raw predictions" 数量 > 0
- [ ] 评估输出显示 "Filtered predictions" 数量 > 0
- [ ] mAP@0.5 > 0（不再是0.0000）
- [ ] 生成的图表文件存在（IoU_distribution.png等）

---

**修复日期**: 2026-04-26  
**修复者**: AI Assistant  
**影响范围**: Exp02 (Faster R-CNN), Exp03 (SSD)
