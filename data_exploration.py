import os
import nltk
import numpy as np
import skimage.io as io
from collections import Counter
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from utils import get_loader
import torch
from torchvision import transforms

plt.switch_backend('TKAgg')


############################################################
# 1、探索COCO数据集的图片
############################################################

# 初始化 COCO API for instance annotations
dataDir = './dataset/annotations_trainval2014/'
dataType = 'val2014'
instances_annFile = os.path.join(dataDir, 'annotations/instances_{}.json'.format(dataType))
coco = COCO(instances_annFile)

# 初始化 COCO API for caption annotations
captions_annFile = os.path.join(dataDir, 'annotations/captions_{}.json'.format(dataType))
coco_caps = COCO(captions_annFile)

# get image ids
ids = list(coco.anns.keys())

# 从数据集中随机选择一张图像，并为其绘图，以及五个相应的标注
ann_id = np.random.choice(ids)
img_id = coco.anns[ann_id]['image_id']
img = coco.loadImgs(img_id)[0]
url = img['coco_url']

# 打印图片URL和可视化对应图片
print(url)
I = io.imread(url)
plt.axis('off')
plt.imshow(I)
plt.show()

# load and display captions
annIds = coco_caps.getAnnIds(imgIds=img['id']);
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)


############################################################
# 2、探索图片的标注文本数据
############################################################

# Set the minimum word count threshold.
# vocab_threshold 指在将单词用作词汇表的一部分之前，单词必须出现在训练图像标注中的总次数。在训练图像标注中出现少于vocab_threshold 的单词将被认为是未知单词
# 如果token在训练集中出现的次数不小于vocab_threshold次数，则将其作为键添加到该字典中并分配一个相应的唯一整数
# 注意：较小的vocab_threshold值会在词汇表中生成更多的token
vocab_threshold = 5

# 指定 batch size.
batch_size = 16

# Define a transform to pre-process the training images.
transform_train = transforms.Compose([
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# 数据加载器get_loader()
# vocab_threshold：如果token在训练集中出现的次数不小于vocab_threshold次数，则将其作为键添加到该字典中并分配一个相应的唯一整数
# vocab_from_file：初次使用，设置成False，运行后会生成“vocab.pkl”文件。后面使用可以设置为True，从“vocab.pkl”文件中加载词汇表，节省时间。
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=False)

# 注意： 如果vocab_from_file=True，则在实例化数s据加载器时为vocab_threshold提供的任何参数都将被完全忽略。
# 如果你要调整vocab_threshold参数的值，则必须设置为vocab_from_file=False，这样才能使更改生效。
# data_loader = get_loader(transform=transform_train,
#                          mode='train',
#                          batch_size=batch_size,
#                          vocab_threshold=vocab_threshold,
#                          vocab_from_file=True)


# 标注预处理
sample_caption = 'A person doing a trick on a rail while riding a skateboard.'
sample_tokens = nltk.tokenize.word_tokenize(str(sample_caption).lower())

# 初始化一个空列表并附加两个整数来分别标记一个图像标注的开头和结尾
sample_caption = []

# 字典中特殊键1：起始单词（"<start>"）是在实例化数据加载器时确定的，并作为参数（start_word）传递
# 整数0始终用于标记一个标注的开头
start_word = data_loader.dataset.vocab.start_word
print('Special start word:', start_word)

# 字典中特殊键2：结束单词（"<end>"）会在实例化数据加载器时被确定，并作为参数（end_word）传递
# 整数1始终用于标记一个标注的结尾
end_word = data_loader.dataset.vocab.end_word
print('Special end word:', end_word)

# 字典中特殊键3：未知的单词（"<unk>"）
unk_word = data_loader.dataset.vocab.unk_word
print('Special unknown word:', unk_word)

sample_caption.append(data_loader.dataset.vocab(start_word))
print('sample_caption:',sample_caption)

sample_caption.extend([data_loader.dataset.vocab(token) for token in sample_tokens])
print('sample_caption:',sample_caption)

sample_caption.append(data_loader.dataset.vocab(end_word))
print('sample_caption:',sample_caption)
# 将整数列表转换为PyTorch张量并将其转换为 long 类型
sample_caption = torch.Tensor(sample_caption).long()
print('sample_caption:',sample_caption)

# word2idx实例变量是一个Python 字典 ，它由字符串值键索引，而这些字符串值键主要是从训练标注获得的token。
# 对于每个键，对应的值是token在预处理步骤中映射到的整数
dict(list(data_loader.dataset.vocab.word2idx.items())[:10])
# 输出键总数 Print the total number of keys in the word2idx dictionary.
print('Total number of tokens in vocabulary:', len(data_loader.dataset.vocab))


############################################################
#  3、使用数据加载器获取批量数据
############################################################

# 查看输出每个长度的训练数据中的标注总数
# Tally the total number of training captions with each length.
counter = Counter(data_loader.dataset.caption_lengths)
lengths = sorted(counter.items(), key=lambda pair: pair[1], reverse=True)
for value, count in lengths:
    print('value: %2d --- count: %5d' % (value, count))

print('All unknown words are mapped to this integer:', data_loader.dataset.vocab(unk_word))

# CoCoDataset类中的get_train_indices方法首先对标注长度进行采样，然后对与训练数据点对应的batch_size indices进行采样，
# 并使用该长度的标注。 这些indices存储在indices下方。
# 这些indices会提供给数据加载器，然后用于检索相应的数据点。该批次中的预处理图像和标注存储在images和captions中
indices = data_loader.dataset.get_train_indices() # Randomly sample a caption length, and sample indices with that length.
print('sampled indices:', indices)
# Obtain the batch.
images, captions = next(iter(data_loader))

print('images.shape:', images.shape)
print('captions.shape:', captions.shape)
# print('images:', images)
print('captions:', captions)


