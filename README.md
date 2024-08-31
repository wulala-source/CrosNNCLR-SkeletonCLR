# CrosNNCLR-SkeletonCLR

Traditional self-supervised contrastive learning approaches regard different views of the same skeleton sequence as a positive pair for the contrastive loss. While existing methods exploit cross-modal retrieval
algorithm of the same skeleton sequence to select positives. The common idea in these work is the following: ignore using other views after data augmentation to obtain more positives. Therefore, we propose a novel and generic Cross-View Nearest Neighbor Contrastive Learning framework for self-supervised action Representation (CrosNNCLR) at the view-level, which can be flexibly integrated into contrastive learning
networks in a plug-and-play manner. CrosNNCLR utilizes different views of skeleton augmentation to obtain the nearest neighbors from features in latent space and consider them as positives embeddings. Extensive experiments on NTU RGB+D 60/120 and PKU-MMD datasets have shown that our CrosNNCLR can outperform previous state-of-the-art methods. Specifically, when equipped with CrosNNCLR, the performance of SkeletonCLR and AimCLR is improved by 0.4%-12.3% and 0.3%-1.9%, respectively.

## Installation
  ```bash
  # Install torchlight
  $ cd torchlight
  $ python setup.py install
  $ cd ..
  # Install other python libraries
  $ pip install -r requirements.txt
  ```
## Data Preparation
- 下载数据集 [NTU RGB+D](https://github.com/shahroudy/NTURGB-D) 
- python tools/ntu60_gendata.py 
- 获取50帧骨架序列：python feeder/preprocess_ntu.py

  
## Acknowledgement
- 本文代码框架是从以下论文代码中扩展的，非常感谢作者们发布这些代码。
- 代码框架是基于 [CrosSCLR](https://github.com/LinguoLi/CrosSCLR).
- 编码器网络基于 [ST-GCN](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md).
