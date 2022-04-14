import torch
import torch.nn as nn

from data import n_categories, n_letters, letterToTensor, lineToTensor, device, all_categories


class RNN(nn.Module):
    # 初始化定义每一层的输入大小和输出大小
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()  # 继承父类的init方法

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # 设置网络中的全连接层
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    # 前向传播过程
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)  # 按维数1拼接（横着拼）
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    # 初始化隐藏层状态 h0
    def initHidden(self):
        return torch.zeros(1, self.hidden_size).to(device)


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
rnn = rnn.to(device)

# 输入字母A测试
input = letterToTensor('A')
hidden = torch.zeros(1, n_hidden).to(device)

output, next_hidden = rnn(input, hidden)
print(output)

# 输入名字Albert的第一个字母A测试
input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden).to(device)

output, next_hidden = rnn(input[0], hidden)
print(output)

def categoryFromOutput(output):  #定义一个函数把y转换为对应的类别，用Tensor.topk选出概率最大的那个概率的下标category_i，就是y的类别
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

print(categoryFromOutput(output))