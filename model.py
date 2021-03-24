import torch
import torch.nn as nn
import torchvision.models as models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################################
# 1、定义EncoderCNN
############################################################

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
    
############################################################
# 2、定义DecoderRNN
############################################################

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=self.hidden_size,
                            num_layers=1, dropout=0, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, vocab_size)

    def init_hidden(self, batch_size):
        return (torch.zeros((1, batch_size, self.hidden_size), device=device),
                torch.zeros((1, batch_size, self.hidden_size), device=device))

    def forward(self, features, captions):
        # features.shape: [batch_size,embed_size]
        batch_size = features.shape[0]
        # 不要 <end>
        # 新的captions.shape: [batch_size, seq_length-1]
        captions = captions[:, :-1]
        # embeddings.shape: [batch_size, seq_length-1, embed_size]
        embeddings = self.embeddings(captions)
        self.hidden = self.init_hidden(batch_size)
        # 连接CNN提取的img特征向量和caption的embeded
        # features.unsqueeze(1).shape: [batch_size, 1, embed_size]
        # concat_vector.shape: [batch_size, seq_length, embed_size]
        concat_vector = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        # out.shape: [batch_size, seq_length, hidden_size]
        out, self.hidden = self.lstm(concat_vector, self.hidden)
        # out.shape: [batch_size, seq_length, vocab_size]
        out = self.fc(out)
        return out

    def sample(self, inputs):
        """
        :param inputs:对应于单个图像的嵌入输入特征的PyTorch张量features
        :return:返回一个Python列表output，用于指示预测的语句。output[i]是一个非负整数，用于标识句子中预测的第i个标记
        """
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        output = []
        batch_size = inputs.shape[0]
        hidden = self.init_hidden(batch_size)

        while True:
            out, hidden = self.lstm(inputs, hidden)
            out = self.fc(out)
            out = out.squeeze(1)
            _, max_indice = torch.max(out, dim=1)
            output.append(max_indice.cpu().numpy()[0].item())

            if (max_indice == 1):
                break

            inputs = self.embeddings(max_indice)
            inputs = inputs.unsqueeze(1)

        return output
