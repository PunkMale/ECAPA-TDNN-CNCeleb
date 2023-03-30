# 介绍
这个项目基于 [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN) 所修改。

## 性能
> 代码还未完成，修改中...

# Quick Start

## 准备工作
此项目针对 [CN-Celeb](http://cnceleb.org/) 数据集进行训练测试。

### 数据
* CN-Celeb 1  [[点此下载]](http://openslr.org/82/)
* CN-Celeb 2  [[点此下载]](http://openslr.org/82/)
> CN-Celeb 原始数据是 flac 格式，考虑到转换格式又要占磁盘空间，就直接读取 flac 进行训练了。

### 环境

## 训练
1. 在`trainECAPAModel.py`配置好对应路径
2. 运行`python trainECAPAModel.py`

## 测试
运行`python trainECAPAModel.py --eval`

## Acknowledge

参考了以下这些项目，感谢各位前辈！

[TaoRuijie/ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)

[Lantian Li/Sunine](https://gitlab.com/csltstu/sunine)
