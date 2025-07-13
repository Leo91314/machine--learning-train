#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速MNIST数据下载器
提供最简单的方式获取MNIST数据集
"""

import urllib.request
import gzip
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def download_mnist():
    """
    从Yann LeCun网站下载MNIST数据集
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
    
    print("正在下载MNIST数据集...")
    
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
                return None
        else:
            print(f"✓ {filename} 已存在")
    
    return data_dir

def load_mnist(data_dir):
    """
    加载MNIST数据
    """
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            f.read(16)  # 跳过头部
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(-1, 28, 28)
    
    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            f.read(8)  # 跳过头部
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data
    
    data_dir = Path(data_dir)
    
    print("正在加载数据...")
    train_images = load_images(data_dir / 'train-images-idx3-ubyte.gz')
    train_labels = load_labels(data_dir / 'train-labels-idx1-ubyte.gz')
    test_images = load_images(data_dir / 't10k-images-idx3-ubyte.gz')
    test_labels = load_labels(data_dir / 't10k-labels-idx1-ubyte.gz')
    
    print("✓ 数据加载完成")
    return train_images, train_labels, test_images, test_labels

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
    主函数
    """
    print("=" * 40)
    print("MNIST数据集下载器")
    print("=" * 40)
    
    # 下载数据
    data_dir = download_mnist()
    
    if data_dir is None:
        print("下载失败，请检查网络连接")
        return
    
    # 加载数据
    train_images, train_labels, test_images, test_labels = load_mnist(data_dir)
    
    # 显示数据信息
    print(f"\n数据集信息:")
    print(f"训练图像: {train_images.shape} (60,000张28x28图像)")
    print(f"训练标签: {train_labels.shape} (60,000个标签)")
    print(f"测试图像: {test_images.shape} (10,000张28x28图像)")
    print(f"测试标签: {test_labels.shape} (10,000个标签)")
    
    # 显示样本
    show_samples(train_images, train_labels)
    
    print(f"\n数据已保存到: {data_dir}")
    print("现在你可以使用这些数据进行机器学习实验了！")

if __name__ == "__main__":
    main() 