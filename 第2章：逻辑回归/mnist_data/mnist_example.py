#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST数据集使用示例
展示如何使用MNIST数据进行机器学习实验
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pickle

def load_mnist_data(filename='mnist_data.pkl'):
    """
    加载MNIST数据
    """
    print("正在加载MNIST数据...")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    train_images = data['train_images']
    train_labels = data['train_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']
    
    print(f"数据加载完成:")
    print(f"训练图像: {train_images.shape}")
    print(f"训练标签: {train_labels.shape}")
    print(f"测试图像: {test_images.shape}")
    print(f"测试标签: {test_labels.shape}")
    
    return train_images, train_labels, test_images, test_labels

def preprocess_data(train_images, test_images):
    """
    数据预处理
    """
    print("正在进行数据预处理...")
    
    # 归一化到0-1范围
    train_images_norm = train_images.astype('float32') / 255.0
    test_images_norm = test_images.astype('float32') / 255.0
    
    # 重塑为2D格式（用于传统机器学习）
    train_images_2d = train_images_norm.reshape(-1, 28*28)
    test_images_2d = test_images_norm.reshape(-1, 28*28)
    
    print(f"预处理完成:")
    print(f"训练特征: {train_images_2d.shape}")
    print(f"测试特征: {test_images_2d.shape}")
    
    return train_images_2d, test_images_2d

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    训练随机森林模型
    """
    print("\n开始训练随机森林模型...")
    
    # 为了演示，使用部分数据
    sample_size = 10000
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_train_sample = X_train[indices]
    y_train_sample = y_train[indices]
    
    print(f"使用 {sample_size} 个样本进行训练...")
    
    # 训练模型
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_sample, y_train_sample)
    
    # 预测
    y_pred = rf.predict(X_test)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.4f}")
    
    return rf, accuracy

def visualize_predictions(test_images, test_labels, predictions, num_samples=10):
    """
    可视化预测结果
    """
    plt.figure(figsize=(15, 6))
    
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(test_images[i], cmap='gray')
        title = f'True: {test_labels[i]}\nPred: {predictions[i]}'
        color = 'green' if test_labels[i] == predictions[i] else 'red'
        plt.title(title, color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ 保存了{num_samples}个预测结果到 mnist_predictions.png")

def analyze_errors(test_images, test_labels, predictions, num_errors=5):
    """
    分析预测错误的样本
    """
    # 找到预测错误的样本
    error_indices = np.where(test_labels != predictions)[0]
    
    if len(error_indices) == 0:
        print("没有预测错误的样本！")
        return
    
    print(f"\n分析前{min(num_errors, len(error_indices))}个预测错误的样本:")
    
    plt.figure(figsize=(15, 3))
    
    for i in range(min(num_errors, len(error_indices))):
        idx = error_indices[i]
        plt.subplot(1, num_errors, i + 1)
        plt.imshow(test_images[idx], cmap='gray')
        title = f'True: {test_labels[idx]}\nPred: {predictions[idx]}'
        plt.title(title, color='red')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_errors.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ 保存了{min(num_errors, len(error_indices))}个错误样本到 mnist_errors.png")

def main():
    """
    主函数
    """
    print("=" * 50)
    print("MNIST数据集使用示例")
    print("=" * 50)
    
    # 加载数据
    train_images, train_labels, test_images, test_labels = load_mnist_data()
    
    # 数据预处理
    X_train, X_test = preprocess_data(train_images, test_images)
    
    # 训练模型
    model, accuracy = train_random_forest(X_train, train_labels, X_test, test_labels)
    
    # 预测
    predictions = model.predict(X_test)
    
    # 可视化预测结果
    visualize_predictions(test_images, test_labels, predictions)
    
    # 分析错误
    analyze_errors(test_images, test_labels, predictions)
    
    # 详细分类报告
    print("\n详细分类报告:")
    print(classification_report(test_labels, predictions))
    
    print("\n" + "=" * 50)
    print("实验完成！")
    print("=" * 50)
    print(f"最终准确率: {accuracy:.4f}")
    print("生成的文件:")
    print("- mnist_predictions.png: 预测结果可视化")
    print("- mnist_errors.png: 错误样本分析")

if __name__ == "__main__":
    main() 