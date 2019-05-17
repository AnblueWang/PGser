# 基于CMU-DoG数据集的对话系统

## Overview

数据来源于:
>A Dataset for Document Grounded Conversations. *Kangyan Zhou, Shrimai Prabhumoye, Alan W Black*. EMNLP 2018. [arXiv](https://arxiv.org/pdf/1809.07358.pdf)

模型是在pointer-generator的基础上加了个对于knowledge的attention机制。

## requirements
pytorch=1.0  
pandas  
unicodedata  
itertools

## 文件结构
下载CMU-DoG数据集放入当前文件夹。
./Conversations表示对话数据。  
./WikiData表示每部电影的数据描述。  
可以使用preprocess.py将Conversations文件夹中的json数据改为csv形式存入新的文件，分别存在./data/train ./data/valid与./data/test中。  
./data/pointerKnow文件夹保存模型与生成的结果。  

## 运行
进入pointerKnow文件夹python main.py即可
