# 数据集中没有任何标点符号，即不用考虑去掉停用词（出现频率太高的词，如逗号，句号等等）

import numpy as np
import math
import random
import time
from collections import Counter

words_corpus = []  # 语料 ：存储所有的词，不管重复与否
words_diff = []  # 词汇表
train_corpus = []   # 训练语料库
word_times = {}  # 词汇表以及每个词对应的词频
word_vec = {}  # 词汇表词向量，字典类型
word_vec_help = {}  # 词的辅助向量，即syn1neg
m_length = 100  # 词向量的维度
alpha = 0.05  # 学习率   CBOW模式下默认是0.05
window_length = 3  # 窗口大小
min_count = 5  # 最小词频阈值，低于这个频率的词会被移除词汇表
EXP_TABLE_SIZE = 1000   # 摘自源码
MAX_EXP = 6   # 摘自源码
sigmoid_list = []   # 存放sigmoid值
count_word_len = {}  # 统计词汇表的每一个Lk,即len(wj)之和
NEG = 5  # NEG(w)的大小
distance_word_len = {}  # 记录每个Lk的区间大小，只保存最后一个值
per_distance = {}  # 记录每一个单词的距离大小
NEG_M = 100000000  # M的大小
table_NEG = []   # 存放负采样表的初始化概率


# 预存sigmoid值
def sigmoid():
    print("sigmoid")
    for index in range(EXP_TABLE_SIZE):
        x = np.exp((index / EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        sigmoid_list.append(x / (x + 1))


# 统计词频，去重，筛选最小词阈值以及高频词（停用词）
def WordCounter():
    print("WordCounter")

    print("读入数据集")
    filename = "/home/hechaoqun/train_data/data/word2vec/train_corpus.txt"   # 读入数据集
    fr = open(filename, 'r', encoding='UTF-8')
    wordline = fr.readline()   # 逐行读取
    while wordline:
        print("读取数据集")
        for item in wordline.strip().split():
            words_corpus.append(item)  # 语料库
        wordline = fr.readline()

    print("语料库大小：", len(words_corpus))

    stopwords = []   # 停用词
    print("读取停用词")
    filename_stop = "./english"
    fr_stop = open(filename_stop, 'r', encoding='UTF-8')
    wordline_stop = fr_stop.readline()
    while wordline_stop:
        for item in wordline_stop.strip().split():
            stopwords.append(item)
        wordline_stop = fr_stop.readline()
    print("停用词数量:", len(stopwords))

    train_corpus_bef = []
    print("词汇表去低词频")
    word_times_bef = Counter(words_corpus)   # 返回字典，key：无序的word， value：词频
    for item in words_corpus:
        if word_times_bef[item] > min_count:
            train_corpus_bef.append(item)
    print("去除停用词")
    for item in train_corpus_bef:
        if item not in stopwords:
            train_corpus.append(item)
    
    print("生成词汇表")
    for key in word_times_bef.keys():
        if word_times_bef[key] > min_count and key not in stopwords:
            words_diff.append(key)   # 无重复的词汇表
            word_times[key] = word_times_bef[key]   # 记录词与其对应的词频
            print(len(words_diff), key, word_times_bef[key])

    print("语料库原始大小：", len(words_corpus))  # 语料库大小
    print("词汇表大小：", len(words_diff))  # 词汇表
    print("训练语料库大小", len(train_corpus))
    pass

# 词向量初始化（随机），辅助向量初始化（为0）
def initwords():
    print("initwords")
    for item in words_diff:
        word_vec[item] = np.random.random((1, m_length))[0] / m_length
        # 代表生成 1 行 m_length 列的浮点数，浮点数都是从0-1中随机。
        # 为什么要在这里加个0呢？
        '''
            那是因为np.random.random() 生成的是一个多维矩阵，即使是一维的，那也是[[1, 2, 3]].
            所以加个[0]，取出第一行的数据
        '''
        word_vec_help[item] = np.zeros(m_length)  # 每一个词向量的辅助向量初始化
    pass


# 得到2*window_length 个上下文词
def get_windows(center):
    res_context = []   # 存放窗口词

    for index in range(1, len(train_corpus) - 1):  # 前面的词
        i = center - index
        if i >= 0:
            res_context.append(train_corpus[i])

        if len(res_context) == 2 * window_length:
            break

        j = center + index
        if j <= len(train_corpus) - 1:
            res_context.append(train_corpus[j])

        if len(res_context) == 2 * window_length:
            break

    return res_context  # 返回的是word 列表


# 对负采样的词汇表进行带权初始化
def initweight():
    print("initweight")
    sum = 0
    for item in words_diff:
        sum = sum + math.pow(word_times[item], 0.75)  # 计算词频之和，即count(u)3/4的sum

    # 计算每个Lk大小，方便计算区间大小
    for item in words_diff:
        per_distance[item] = math.pow(word_times[item], 0.75) / sum  # 计算每一个词的len(w)

    pass


#生成负采样的概率表
def init_NEG():
    print("init_NEG")
    # 对词频进行排序（从小到大）
    wordtime = sorted(word_times.items(), key=lambda x: x[1], reverse=False)
    #   排序返回的是 包含元组的列表   [('word', 1), ('supporting', 1), ......,('conduct', 1)]
    i = 0
    d1 = per_distance[wordtime[i][0]]
    for a in range(NEG_M):
        table_NEG.append(wordtime[i][0])
        if (a / NEG_M) > d1:
            i = i + 1
            d1 = d1 + per_distance[wordtime[i][0]]

        if i >= len(wordtime):
            i = len(wordtime) - 1

    pass


# 选择负样本词
def choice_NEG(target_word):

    list_NEG = []
    while len(list_NEG) < NEG:   # 5个词
        m = random.randrange(0, NEG_M)
        if table_NEG[m] != target_word:
            list_NEG.append(table_NEG[m])

    list_NEG.append(target_word)

    return list_NEG

'''
def sigmoid(inx):
    #print("sigmoid")
    if inx >= 0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0 /(1 + np.exp(-inx))
    else:
        return np.exp(inx)/(1 + np.exp(inx))
'''

def CB_NS():
    print("CB_NS")
    
    global alpha

    for index in range(len(train_corpus)):  # 开始选词，找窗口

        print("窗口", index)
        
        #更新学习率
        if index != 0 and index % 10000 == 0:
            if alpha < 0.0001:
                alpha = 0.0001
            else:
                alpha = alpha * (1 - (index / (len(train_corpus) + 1)))
        
        context = get_windows(index)  # 找窗口
        

        word_center = train_corpus[index]  # 中心词
        NEG_words = choice_NEG(word_center)  # 负样本词和中心词的集合

        global neule  # e
        neule = np.zeros(m_length)

        Xw = np.zeros(m_length)

        for item1 in context:
            Xw = Xw + word_vec[item1]  # 词向量之和

        for item in NEG_words:

            x_vec = np.dot(Xw, word_vec_help[item])
            if x_vec > MAX_EXP:
                q = 1
            elif x_vec < -MAX_EXP:
                q = 0
            else:
                sigmoid_index = (x_vec + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2)
                q = sigmoid_list[int(sigmoid_index)]

            if item == word_center:
                Lw = 1
            else:
                Lw = 0
            g = alpha * (Lw - q)

            neule = neule + g * word_vec_help[item]

            word_vec_help[item] = word_vec_help[item] + g * Xw

        for item2 in context:
            word_vec[item2] = word_vec[item2] + neule

    pass


if __name__ == '__main__':
    start = time.time()
    print("程序正在执行......")

    sigmoid()
    WordCounter()  # 对数据集进行处理
    initwords()  # 对词向量初始化
    initweight()   # 初始化负采样表的第一步
    init_NEG()   # 初始化负采样表的第二步


    CB_NS()   #开始训练

    end = time.time()
    print("程序运行时间:" + str(end - start) + "s")
