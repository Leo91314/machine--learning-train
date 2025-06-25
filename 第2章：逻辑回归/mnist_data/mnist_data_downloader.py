#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST数据集下载器
提供多种方式获取MNIST手写数字数据集
"""

import os
import urllib.request
import gzip
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def download_mnist_manual():
    """
    手动下载MNIST数据集（从Yann LeCun的网站）
    """
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz', 
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    # 创建数据目录
    data_dir = Path("mnist_data")
    data_dir.mkdir(exist_ok=True)
    
    print("开始下载MNIST数据集...")
    
    for name, filename in files.items():
        url = base_url + filename
        filepath = data_dir / filename
        
        if not filepath.exists():
            print(f"下载 {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"✓ {filename} 下载完成")
            except Exception as e:
                print(f"✗ 下载 {filename} 失败: {e}")
        else:
            print(f"✓ {filename} 已存在")
    
    return data_dir

def load_mnist_manual(data_dir):
    """
    加载手动下载的MNIST数据
    """
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            # 跳过前16字节的头部信息
            f.read(16)
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(-1, 28, 28)
    
    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            # 跳过前8字节的头部信息
            f.read(8)
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data
    
    data_dir = Path(data_dir)
    
    # 加载数据
    train_images = load_images(data_dir / 'train-images-idx3-ubyte.gz')
    train_labels = load_labels(data_dir / 'train-labels-idx1-ubyte.gz')
    test_images = load_images(data_dir / 't10k-images-idx3-ubyte.gz')
    test_labels = load_labels(data_dir / 't10k-labels-idx1-ubyte.gz')
    
    return train_images, train_labels, test_images, test_labels

def download_mnist_torch():
    """
    使用torchvision下载MNIST数据
    """
    try:
        import torch
        import torchvision
        from torchvision import datasets, transforms
        
        print("使用torchvision下载MNIST数据...")
        
        # 定义数据转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
        ])
        
        # 下载训练集
        train_dataset = datasets.MNIST(
            root='./mnist_torch', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        # 下载测试集
        test_dataset = datasets.MNIST(
            root='./mnist_torch', 
            train=False, 
            download=True, 
            transform=transform
        )
        
        print("✓ torchvision MNIST数据下载完成")
        return train_dataset, test_dataset
        
    except ImportError:
        print("✗ torchvision未安装，请运行: pip install torch torchvision")
        return None, None

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
        return (train_images, train_labels), (test_images, test_labels)
        
    except ImportError:
        print("✗ TensorFlow未安装，请运行: pip install tensorflow")
        return None, None

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

def visualize_mnist(train_images, train_labels, num_samples=5):
    """
    可视化MNIST数据
    """
    plt.figure(figsize=(15, 3))
    
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(train_images[i], cmap='gray')
        plt.title(f'Label: {train_labels[i]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ 保存了{num_samples}个MNIST样本图像到 mnist_samples.png")

def main():
    """
    主函数：演示不同的MNIST数据获取方式
    """
    print("=" * 50)
    print("MNIST数据集下载器")
    print("=" * 50)
    
    # 方式1：手动下载（从Yann LeCun网站）
    print("\n1. 手动下载方式（从Yann LeCun网站）")
    print("-" * 30)
    data_dir = download_mnist_manual()
    
    if data_dir.exists():
        train_images, train_labels, test_images, test_labels = load_mnist_manual(data_dir)
        print(f"训练图像形状: {train_images.shape}")
        print(f"训练标签形状: {train_labels.shape}")
        print(f"测试图像形状: {test_images.shape}")
        print(f"测试标签形状: {test_labels.shape}")
        
        # 可视化一些样本
        visualize_mnist(train_images, train_labels)
    
    # 方式2：使用torchvision
    print("\n2. 使用torchvision下载")
    print("-" * 30)
    train_dataset, test_dataset = download_mnist_torch()
    
    # 方式3：使用TensorFlow
    print("\n3. 使用TensorFlow下载")
    print("-" * 30)
    tf_data = download_mnist_tensorflow()
    
    # 方式4：使用scikit-learn
    print("\n4. 使用scikit-learn下载")
    print("-" * 30)
    sk_data = download_mnist_sklearn()
    
    print("\n" + "=" * 50)
    print("下载完成！")
    print("=" * 50)
    print("\n数据文件位置:")
    print(f"- 手动下载: {data_dir}")
    print("- torchvision: ./mnist_torch/")
    print("- TensorFlow: 内存中")
    print("- scikit-learn: 内存中")
    
    print("\n使用建议:")
    print("1. 如果使用PyTorch，推荐使用torchvision方式")
    print("2. 如果使用TensorFlow，推荐使用TensorFlow方式")
    print("3. 如果需要快速原型，推荐使用scikit-learn方式")
    print("4. 手动下载方式适用于所有情况，但需要自己处理数据格式")

if __name__ == "__main__":
    main() 