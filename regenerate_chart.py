#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新生成图表（使用英文标签）
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取结果
if os.path.exists("results_summary.csv"):
    results_df = pd.read_csv("results_summary.csv", encoding='utf-8-sig')
    
    # 更新模型名称为英文（如果还是中文的话）
    if "初始模型" in results_df['模型'].values[0]:
        results_df['模型'] = results_df['模型'].replace({
            "初始模型（加载）": "Initial Model",
            "策略1（继续训练）": "Strategy 1 (Continue Training)",
            "策略2（调整学习率）": "Strategy 2 (Adjusted LR)"
        })
    
    # 可视化
    print("正在生成可视化图表（英文标签）...")
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
else:
    print("错误: 找不到 results_summary.csv 文件")

