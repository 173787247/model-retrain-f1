#!/usr/bin/env python
"""
执行模型加载、评估和再训练以提高 F1 Score
这个脚本会执行 notebook 中的训练代码
"""
import sys
import os
import time
from datetime import datetime

print("=" * 80)
print("模型加载、评估和再训练以提高 F1 Score")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# 导入必要的库
try:
    from datasets import load_dataset
    import evaluate
    from transformers import (
        AutoTokenizer,
        AutoModelForQuestionAnswering,
        TrainingArguments,
        Trainer,
        default_data_collator
    )
    import torch
    import numpy as np
    import collections
    from tqdm.auto import tqdm
    import pandas as pd
    import matplotlib.pyplot as plt
    import transformers
    print("✓ 所有依赖库已导入")
except ImportError as e:
    print(f"✗ 导入错误: {e}")
    print("正在安装依赖...")
    os.system("pip install -q transformers datasets accelerate evaluate matplotlib pandas tqdm")
    from datasets import load_dataset
    import evaluate
    from transformers import (
        AutoTokenizer,
        AutoModelForQuestionAnswering,
        TrainingArguments,
        Trainer,
        default_data_collator
    )
    import torch
    import numpy as np
    import collections
    from tqdm.auto import tqdm
    import pandas as pd
    import matplotlib.pyplot as plt
    import transformers

# 检查 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 配置参数
squad_v2 = False
model_checkpoint = "distilbert-base-uncased"
batch_size = 16
model_dir = "models/distilbert-base-uncased-finetuned-squad"

print(f"\n模型检查点: {model_checkpoint}")
print(f"批次大小: {batch_size}")
print(f"本地模型路径: {model_dir}")

# 1. 加载数据集
print("\n" + "=" * 80)
print("步骤 1: 加载数据集")
print("=" * 80)
print("正在加载 SQuAD 数据集...")
datasets = load_dataset("squad_v2" if squad_v2 else "squad")
print("✓ 数据集加载完成！")
print(f"  训练集大小: {len(datasets['train']):,}")
print(f"  验证集大小: {len(datasets['validation']):,}")

# 2. 数据预处理
print("\n" + "=" * 80)
print("步骤 2: 数据预处理")
print("=" * 80)
print(f"正在加载分词器: {model_checkpoint}...")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

max_length = 384
doc_stride = 128
pad_on_right = tokenizer.padding_side == "right"

print(f"最大长度: {max_length}")
print(f"文档步长: {doc_stride}")

# 预处理函数（从notebook复制）
def prepare_train_features(examples):
    examples["question"] = [q.lstrip() for q in examples["question"]]
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1
            
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
    
    return tokenized_examples

def prepare_validation_features(examples):
    examples["question"] = [q.lstrip() for q in examples["question"]]
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []
    
    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]
    
    return tokenized_examples

print("正在预处理训练数据...")
tokenized_datasets = datasets.map(
    prepare_train_features,
    batched=True,
    remove_columns=datasets["train"].column_names,
    num_proc=4
)
print("✓ 训练数据预处理完成！")

print("正在预处理验证数据...")
validation_features = datasets["validation"].map(
    prepare_validation_features,
    batched=True,
    remove_columns=datasets["validation"].column_names,
    num_proc=4
)
print("✓ 验证数据预处理完成！")

# 3. 加载或训练初始模型
print("\n" + "=" * 80)
print("步骤 3: 加载或训练初始模型")
print("=" * 80)
print(f"正在从 {model_dir} 加载模型...")

if os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")):
    print("✓ 找到已保存的模型，正在加载...")
    trained_model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    print("✓ 模型加载成功！")
else:
    print("✗ 未找到已保存的模型，将进行初始训练...")
    print("  这可能需要 1-2 小时...")
    
    trained_model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    data_collator = default_data_collator
    
    args_initial = TrainingArguments(
        output_dir=model_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )
    
    trainer_initial = Trainer(
        trained_model,
        args_initial,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("开始初始训练...")
    trainer_initial.train()
    trainer_initial.save_model(model_dir)
    print("✓ 初始训练完成，模型已保存！")

# 4. 评估初始模型
print("\n" + "=" * 80)
print("步骤 4: 评估初始模型")
print("=" * 80)

# 后处理函数
def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    
    predictions = collections.OrderedDict()
    
    print(f"正在后处理 {len(examples)} 个示例的预测...")
    
    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        valid_answers = []
        context = example["context"]
        
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score
            
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}
        
        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer
    
    return predictions

data_collator = default_data_collator
args = TrainingArguments(
    output_dir=model_dir,
    per_device_eval_batch_size=batch_size,
)

trainer = Trainer(
    trained_model,
    args,
    eval_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("正在对验证集进行预测...")
validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))
raw_predictions = trainer.predict(validation_features)

print("正在后处理预测结果...")
final_predictions = postprocess_qa_predictions(
    datasets["validation"], 
    validation_features, 
    raw_predictions.predictions
)

# 计算指标
metric = evaluate.load("squad_v2" if squad_v2 else "squad")

if squad_v2:
    formatted_predictions = [
        {"id": k, "prediction_text": v, "no_answer_probability": 0.0} 
        for k, v in final_predictions.items()
    ]
else:
    formatted_predictions = [
        {"id": k, "prediction_text": v} 
        for k, v in final_predictions.items()
    ]

references = [
    {"id": ex["id"], "answers": ex["answers"]} 
    for ex in datasets["validation"]
]

initial_metrics = metric.compute(predictions=formatted_predictions, references=references)

print("\n初始模型评估结果:")
print(f"  F1 Score: {initial_metrics['f1']:.4f}")
print(f"  Exact Match: {initial_metrics['exact_match']:.4f}")

initial_f1 = initial_metrics['f1']

# 5. 再训练策略
print("\n" + "=" * 80)
print("步骤 5: 再训练以提高 F1 Score")
print("=" * 80)

all_results = [
    {
        "模型": "Initial Model",
        "F1 Score": initial_f1,
        "Exact Match": initial_metrics['exact_match'],
        "提升": 0.0
    }
]

# 策略1：继续训练
print("\n策略1：继续训练（更多轮数，较低学习率）")
model_strategy1 = AutoModelForQuestionAnswering.from_pretrained(model_dir)

args_strategy1 = TrainingArguments(
    output_dir=f"{model_dir}_strategy1",
    eval_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=2,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
)

trainer_strategy1 = Trainer(
    model_strategy1,
    args_strategy1,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("开始再训练...")
trainer_strategy1.train()
trainer_strategy1.save_model(f"{model_dir}_strategy1")

# 评估策略1
print("正在评估策略1的结果...")
raw_predictions_s1 = trainer_strategy1.predict(validation_features)
final_predictions_s1 = postprocess_qa_predictions(
    datasets["validation"], 
    validation_features, 
    raw_predictions_s1.predictions
)

if squad_v2:
    formatted_predictions_s1 = [
        {"id": k, "prediction_text": v, "no_answer_probability": 0.0} 
        for k, v in final_predictions_s1.items()
    ]
else:
    formatted_predictions_s1 = [
        {"id": k, "prediction_text": v} 
        for k, v in final_predictions_s1.items()
    ]

metrics_s1 = metric.compute(predictions=formatted_predictions_s1, references=references)
strategy1_f1 = metrics_s1['f1']

print(f"策略1结果:")
print(f"  F1 Score: {strategy1_f1:.4f} (提升: {strategy1_f1 - initial_f1:+.4f})")
print(f"  Exact Match: {metrics_s1['exact_match']:.4f}")

all_results.append({
    "模型": "Strategy 1 (Continue Training)",
    "F1 Score": strategy1_f1,
    "Exact Match": metrics_s1['exact_match'],
    "提升": strategy1_f1 - initial_f1
})

# 策略2：调整学习率
print("\n策略2：调整学习率和训练轮数")
model_strategy2 = AutoModelForQuestionAnswering.from_pretrained(model_dir)

args_strategy2 = TrainingArguments(
    output_dir=f"{model_dir}_strategy2",
    eval_strategy="epoch",
    save_strategy="epoch",  # 必须与 eval_strategy 匹配
    learning_rate=5e-6,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # 使用 eval_loss 因为 F1 需要通过后处理计算
    greater_is_better=False,  # loss 越小越好
)

trainer_strategy2 = Trainer(
    model_strategy2,
    args_strategy2,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("开始再训练...")
trainer_strategy2.train()
trainer_strategy2.save_model(f"{model_dir}_strategy2")

# 评估策略2
print("正在评估策略2的结果...")
raw_predictions_s2 = trainer_strategy2.predict(validation_features)
final_predictions_s2 = postprocess_qa_predictions(
    datasets["validation"], 
    validation_features, 
    raw_predictions_s2.predictions
)

if squad_v2:
    formatted_predictions_s2 = [
        {"id": k, "prediction_text": v, "no_answer_probability": 0.0} 
        for k, v in final_predictions_s2.items()
    ]
else:
    formatted_predictions_s2 = [
        {"id": k, "prediction_text": v} 
        for k, v in final_predictions_s2.items()
    ]

metrics_s2 = metric.compute(predictions=formatted_predictions_s2, references=references)
strategy2_f1 = metrics_s2['f1']

print(f"策略2结果:")
print(f"  F1 Score: {strategy2_f1:.4f} (提升: {strategy2_f1 - initial_f1:+.4f})")
print(f"  Exact Match: {metrics_s2['exact_match']:.4f}")

all_results.append({
    "模型": "Strategy 2 (Adjusted LR)",
    "F1 Score": strategy2_f1,
    "Exact Match": metrics_s2['exact_match'],
    "提升": strategy2_f1 - initial_f1
})

# 6. 结果汇总
print("\n" + "=" * 80)
print("所有策略的结果对比")
print("=" * 80)
results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))

best_result = results_df.loc[results_df['F1 Score'].idxmax()]
print("\n" + "=" * 80)
print("最佳 F1 Score 配置:")
print("=" * 80)
print(f"模型: {best_result['模型']}")
print(f"F1 Score: {best_result['F1 Score']:.4f}")
print(f"Exact Match: {best_result['Exact Match']:.4f}")
print(f"相对初始模型提升: {best_result['提升']:+.4f} ({best_result['提升']/initial_f1*100:+.2f}%)")

# 保存结果
results_df.to_csv("results_summary.csv", index=False, encoding='utf-8-sig')
print(f"\n结果已保存到: results_summary.csv")

# 可视化
print("\n正在生成可视化图表...")
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = plt.bar(results_df['模型'], results_df['F1 Score'], color=colors)
plt.title('F1 Score Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Model/Strategy', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)
plt.xticks(rotation=45, ha='right')
for i, (bar, v) in enumerate(zip(bars, results_df['F1 Score'])):
    plt.text(bar.get_x() + bar.get_width()/2, v + 0.005, 
             f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
colors_improve = ['#d62728' if x < 0 else '#2ca02c' for x in results_df['提升']]
bars2 = plt.bar(results_df['模型'], results_df['提升'], color=colors_improve)
plt.title('F1 Score Improvement', fontsize=14, fontweight='bold')
plt.xlabel('Model/Strategy', fontsize=12)
plt.ylabel('F1 Score Improvement', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
for i, (bar, v) in enumerate(zip(bars2, results_df['提升'])):
    plt.text(bar.get_x() + bar.get_width()/2, v + (0.001 if v >= 0 else -0.001), 
             f'{v:+.4f}', ha='center', va='bottom' if v >= 0 else 'top', fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('f1_score_comparison.png', dpi=300, bbox_inches='tight')
print("图表已保存为: f1_score_comparison.png")

print("\n" + "=" * 80)
print(f"训练完成！结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

