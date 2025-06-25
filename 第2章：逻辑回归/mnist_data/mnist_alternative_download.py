#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST数据集替代下载方案
使用机器学习库下载MNIST数据
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

def download_mnist_sklearn():
    """
    使用scikit-learn下载MNIST数据
    """
    try:
        from sklearn.datasets import fetch_openml
        print("使用scikit-learn下载MNIST数据...")
        
        # 下载MNIST数据
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        
        # 分离特征和标签
        X, y = mnist.data, mnist.target
        
        # 重塑为28x28图像
        X = X.reshape(-1, 28, 28)
        
        # 分离训练集和测试集
        train_images = X[:60000]
        train_labels = y[:60000].astype(int)
        test_images = X[60000:]
        test_labels = y[60000:].astype(int)
        
        print("✓ scikit-learn MNIST数据下载完成")
        return train_images, train_labels, test_images, test_labels
        
    except ImportError:
        print("✗ scikit-learn未安装，请运行: pip install scikit-learn")
        return None, None, None, None
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return None, None, None, None

def download_mnist_tensorflow():
    """
    使用TensorFlow下载MNIST数据
    """
    try:
        import tensorflow as tf
        print("使用TensorFlow下载MNIST数据...")
        
        # 下载MNIST数据
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        
        print("✓ TensorFlow MNIST数据下载完成")
        return train_images, train_labels, test_images, test_labels
        
    except ImportError:
        print("✗ TensorFlow未安装，请运行: pip install tensorflow")
        return None, None, None, None
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return None, None, None, None

def download_mnist_torch():
    """
    使用torchvision下载MNIST数据
    """
    try:
        import torch
        import torchvision
        from torchvision import datasets, transforms
        
        print("使用torchvision下载MNIST数据...")
        
        # 下载训练集
        train_dataset = datasets.MNIST(
            root='./mnist_torch', 
            train=True, 
            download=True, 
            transform=None
        )
        
        # 下载测试集
        test_dataset = datasets.MNIST(
            root='./mnist_torch', 
            train=False, 
            download=True, 
            transform=None
        )
        
        # 转换为numpy数组
        train_images = train_dataset.data.numpy()
        train_labels = train_dataset.targets.numpy()
        test_images = test_dataset.data.numpy()
        test_labels = test_dataset.targets.numpy()
        
        print("✓ torchvision MNIST数据下载完成")
        return train_images, train_labels, test_images, test_labels
        
    except ImportError:
        print("✗ torchvision未安装，请运行: pip install torch torchvision")
        return None, None, None, None
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return None, None, None, None

def save_mnist_data(train_images, train_labels, test_images, test_labels, filename='mnist_data.pkl'):
    """
    保存MNIST数据到本地文件
    """
    data = {
        'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ 数据已保存到 {filename}")

def load_mnist_data(filename='mnist_data.pkl'):
    """
    从本地文件加载MNIST数据
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    return (data['train_images'], data['train_labels'], 
            data['test_images'], data['test_labels'])

def show_samples(train_images, train_labels, num_samples=5):
    """
    显示一些MNIST样本
    """
    plt.figure(figsize=(15, 3))
    
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(train_images[i], cmap='gray')
        plt.title(f'数字: {train_labels[i]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ 保存了{num_samples}个样本图像到 mnist_samples.png")

def main():
    """
    主函数：尝试多种方式下载MNIST数据
    """
    print("=" * 50)
    print("MNIST数据集替代下载方案")
    print("=" * 50)
    
    # 尝试不同的下载方式
    methods = [
        ("scikit-learn", download_mnist_sklearn),
        ("TensorFlow", download_mnist_tensorflow),
        ("torchvision", download_mnist_torch)
    ]
    
    train_images = None
    train_labels = None
    test_images = None
    test_labels = None
    
    for method_name, download_func in methods:
        print(f"\n尝试使用 {method_name} 下载...")
        result = download_func()
        
        if result[0] is not None:
            train_images, train_labels, test_images, test_labels = result
            print(f"✓ 使用 {method_name} 成功下载MNIST数据")
            break
        else:
            print(f"✗ {method_name} 下载失败")
    
    if train_images is None:
        print("\n所有下载方式都失败了。请检查网络连接或安装必要的库。")
        print("\n安装命令:")
        print("pip install scikit-learn")
        print("pip install tensorflow")
        print("pip install torch torchvision")
        return
    
    # 显示数据信息
    print(f"\n数据集信息:")
    print(f"训练图像: {train_images.shape} (60,000张28x28图像)")
    print(f"训练标签: {train_labels.shape} (60,000个标签)")
    print(f"测试图像: {test_images.shape} (10,000张28x28图像)")
    print(f"测试标签: {test_labels.shape} (10,000个标签)")
    
    # 显示样本
    show_samples(train_images, train_labels)
    
    # 保存数据
    save_mnist_data(train_images, train_labels, test_images, test_labels)
    
    print(f"\n数据已成功下载并保存！")
    print("现在你可以使用这些数据进行机器学习实验了！")

if __name__ == "__main__":
    main() 