from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch

def findFiles(path): return glob.glob(path)  #glob(path)查询目录下文件

def unicodeToAscii(str):   #记下即可
    return ''.join(
        c for c in unicodedata.normalize('NFD', str)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(findFiles('data/names/*.txt'))   #查找以.txt结尾的文件

all_letters = string.ascii_letters + '.,;'  #ascii_letters生成所有字母
n_letters = len(all_letters)

print(unicodeToAscii('Ślusàrski'))

# 文件命名`[language].txt`中，`language`是类别category，把每个文件打开，
# 存入一个数组`lines = [name1,...]`，建立一个词典`category_lines = {language: lines}`
category_lines = {}
all_categories = []


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')  # 先read读取，strip去除首尾空格，split以'\n'分隔
    return [unicodeToAscii(line) for line in lines]  # 将lines中的name读出放入字母标准化函数并输出标准的name


for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[
        0]  # os.path.basename(path)返回path最后的文件名，os.path.splitext() 将文件名和扩展名分开
    all_categories.append(category)
    lines = readLines(filename)  # readLines一次性读取整个文件；自动将文件内容分析成一个行的列表
    category_lines[category] = lines  # 把language与名字组合放在词典里

n_categories = len(all_categories)
print(all_categories)
print(category_lines['Italian'])

#对字母进行one-hot编码，转成 tensor
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#返回字母letter的索引 index
def letterToIndex(letter):
    return all_letters.find(letter)

#把一个字母编码成tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)     #创建  1 x n_letters 的张量
    #把字母letter 的索引定为1，其他为0
    tensor[0][letterToIndex(letter)] = 1
    return tensor.to(device)

#把一个单词编码为tensor
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    #遍历单词中所有字母，对每个字母letter的索引设为1，其他为0
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor.to(device)

print(letterToTensor('J'))
print(lineToTensor('Jones').size())
