from torchvision import transforms
import os
import sys
import math
import numpy as np
from utils import get_loader
from model import EncoderCNN, DecoderRNN
import torch
import torch.nn as nn
import torch.utils.data as data

############################################################
# 1、训练参数设置
############################################################

batch_size = 256  # batch size
vocab_threshold = 5  # 在训练图像标注中出现少于vocab_threshold 的单词将被认为是未知单词
vocab_from_file = False  # 初次使用，设置`vocab_from_file = False`，运行后会生成“vocab.pkl”文件。
# vocab_from_file = True  # 后面使用可以设置为True，从“vocab.pkl”文件中加载词汇表，节省时间。
embed_size = 1024  # dimensionality of image and word embeddings
hidden_size = 1024  # number of features in hidden state of the RNN decoder
num_epochs = 1  # number of training epochs
save_every = 1  # determines frequency of saving model weights
print_every = 100  # determines window for printing average loss
log_file = 'training_log.txt'  # name of file with saved training loss and perplexity

############################################################
# 2、 设置数据加载器
############################################################

# 图像转换：规定了应该如何对图像进行预处理，并将它们转换为PyTorch张量，然后再将它们用作CNN编码器的输入
transform_train = transforms.Compose([
    transforms.Resize(256),  # smaller edge of image resized to 256
    transforms.RandomCrop(224),  # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
    transforms.ToTensor(),  # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# 构建数据加载器。运行后，数据加载器会存储在变量data_loader。
#'train':（用于批量加载训练数据）或 'test'（用于测试数据）
# vocab_threshold：在将单词用作词汇表的一部分之前，单词必须出现在训练图像标注中的总次数。在训练图像标注中出现少于vocab_threshold 的单词将被认为是未知单词
# vocab_from_file：布尔值，是否从文件中加载词汇表
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold, #
                         vocab_from_file=vocab_from_file)


############################################################
# 3、训练模型
############################################################

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# 初始化 encoder 和 decoder.
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Move models to GPU if CUDA is available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

# 解码器中的所有权重都是可训练的
# 但编码器，只想在嵌入层中训练权重
params = list(decoder.parameters()) + list(encoder.embed.parameters())

optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)

# Set the total number of training steps per epoch.
total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)

# Open the training log file.
f = open(log_file, 'w')


for epoch in range(1, num_epochs + 1):
    for i_step in range(1, total_step + 1):
        # Randomly sample a caption length, and sample indices with that length.
        indices = data_loader.dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader.batch_sampler.sampler = new_sampler

        # Obtain the batch.
        images, captions = next(iter(data_loader))

        # Move batch of images and captions to GPU if CUDA is available.
        images = images.to(device)
        captions = captions.to(device)

        # Zero the gradients.
        decoder.zero_grad()
        encoder.zero_grad()

        # Pass the inputs through the CNN-RNN model.
        features = encoder(images)
        outputs = decoder(features, captions)

        # Calculate the batch loss.
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

        # Backward pass.
        loss.backward()

        # Update the parameters in the optimizer.
        optimizer.step()

        # Get training statistics.
        stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (
        epoch, num_epochs, i_step, total_step, loss.item(), np.exp(loss.item()))

        # Print training statistics (on same line).
        print('\r' + stats, end="")
        sys.stdout.flush()

        # Print training statistics to file.
        f.write(stats + '\n')
        f.flush()

        # Print training statistics (on different line).
        if i_step % print_every == 0:
            print('\r' + stats)

    # Save the weights.
    if epoch % save_every == 0:
        torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-%d.pkl' % epoch))
        torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-%d.pkl' % epoch))

# Close the training log file.
f.close()