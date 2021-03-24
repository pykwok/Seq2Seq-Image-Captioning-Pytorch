## Seq2Seq-Image-Captioning--Pytorch

本项目用COCO数据集训练了一个Seq2Seq模型，实现图片到文字的转换，使模型能够识别图片内容，并根据图片内容生成描述图片的语句。

**Seq2Seq模型结构**：
- Encoder：用预训练好的ResNet101模型，把数据加载器生成出的图片提取出图片特征向量，再接一层word embedding层，把特征向量变成固定形状。
- Decoder：先用word embedding层把数据加载器生成出的图片标注向量转化成固定形状，后接一层LSTM层用于记住图像特征向量里的空间信息和预测下一个单词，最后连上一层全连接层，把维度映射回标注文字向量空间的维度数。

我的配置是：`python 3.7` 和`PyTorch 1.3.1`     
参考论文：[《Show and Tell: A Neural Image Caption Generator》](https://arxiv.org/pdf/1411.4555.pdf)

#### 一、各文件内容：

1. `data_exploration.py`

- 探索COCO数据集的图片数据
- 探索COCO数据集的标注文本数据
- 了解如何使用数据加载器`get_loader()`来获取批量数据

2. `train.py`
- 训练参数设置
- 设置数据加载器
- 训练模型

3. `detect_demo.ipynb`（和`detect.py`内容一样，可视化展示模型结果）
- 设置获取测试数据集的数据加载器
- 加载训练好的模型
- 清理标注，把整数列表转化成文字，输出图片内容

4. `model.py`
- 定义Encoder和Decoder类


#### 二、dataset文件夹结构图
```
├─dataset     
│  ├─annotations     
│  │    captions_train2014.json     
│  │    image_info_test2014.json     
│  ├─annotations_trainval2014     
│  │  └─annotations     
│  │        captions_train2014.json     
│  │        captions_val2014.json     
│  │        instances_train2014.json     
│  │        instances_val2014.json     
│  ├─images    
│  │   ├─test2014     
│  │   └─train2014    
```





