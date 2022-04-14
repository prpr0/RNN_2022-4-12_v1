import random
import torch
import torch.nn as nn
import time
import math

from data import all_categories, category_lines, device, lineToTensor
from model import rnn, categoryFromOutput


def randomChoice(l):
    return l[random.randint(0, len(l)-1)]

def randomTrainingExample():
    category = randomChoice(all_categories)         #采样得到category
    line = randomChoice(category_lines[category])   #从category中采样得到line
    category_tensor = torch.tensor([all_categories.index(category)], dtype = torch.long).to(device)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category = ', category, '/ line = ', line)

    # 定义损失函数 NLLLoss
    criterion = nn.NLLLoss()
    learning_rate = 0.005 #学习率0.005


    def train(category_tensor, line_tensor):
        hidden = rnn.initHidden()

        rnn.zero_grad()

        # RNN的循环
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        loss = criterion(output, category_tensor)
        loss.backward()

        # 更新参数
        for p in rnn.parameters():
            p.data.add_(p.grad.data, alpha=- learning_rate)

        return output, loss.item()

    # 开始训练模型
    n_iters = 100000
    print_every = 5000
    plot_every = 1000

    #记录训练损失
    current_loss = 0
    all_losses = []


    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)


    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0


