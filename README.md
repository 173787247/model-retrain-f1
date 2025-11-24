# 模型加载、评估和再训练以提高 F1 Score

## 作业目标
1. 加载本地保存的模型
2. 对模型进行评估（计算 F1 Score）
3. 再训练模型以提高 F1 Score

## 数据集
使用 SQuAD v1.1 问答数据集进行训练和评估

## 实验策略
1. **初始评估**：加载本地保存的模型，计算初始 F1 Score
2. **策略1**：继续训练（更多轮数，较低学习率）
3. **策略2**：调整学习率和训练轮数（更小的学习率，添加 warmup）

## 文件说明
- `model_retrain_f1.ipynb` - 主 Notebook，包含完整的模型加载、评估和再训练代码

## 运行要求
- Python 3.8+
- PyTorch
- Transformers
- Datasets
- GPU 推荐

## 安装依赖
```bash
pip install transformers datasets accelerate evaluate torch
```

## 使用方法
1. 确保已有训练好的模型保存在 `models/distilbert-base-uncased-finetuned-squad/` 目录
2. 如果没有，notebook 会先进行初始训练
3. 打开 `model_retrain_f1.ipynb`
4. 按顺序运行所有 cell
5. 查看不同策略的 F1 Score 对比

## 结果
训练完成后会生成：
- 初始模型和再训练模型的 F1 Score 对比
- 可视化图表（`f1_score_comparison.png`）
- 最佳 F1 Score 配置信息

