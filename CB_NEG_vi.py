# 数据集中没有任何标点符号，即不用考虑去掉停用词（出现频率太高的词，如逗号，句号等等）

import numpy as np
import math
import random
import time
from collections import Counter

words_corpus = []    # 语料 ：存储所有的词，不管重复与否
words_diff = []   #词汇表
word_times = {}  # 词汇表以及每个词对应的词频
word_vec = {}   # 词汇表词向量，字典类型
word_vec_help = {}   # 词的辅助向量，即syn1neg
m_length = 100   # 词向量的维度
p_learning = 0.05   # 学习率   CBOW模式下默认是0.05
window_length = 3   # 窗口大小
min_time = 3   # 最⼩词频阈值（目前没有用到）
count_word_len = {}   # 统计词汇表的每一个Lk,即len(wj)之和
NEG = 5   # NEG(w)的大小
distance_word_len = {}   #记录每个Lk的区间大小，只保存最后一个值
per_distance = {}   # 记录每一个单词的距离大小
NEG_M= 100000000   # M的大小
min_count = 5   # 最小词频阈值，低于这个频率的词会被移除词汇表
# 不筛选的话，时间太长了


# 统计词频，统计无重复单词，同时去除词频过小的词
def WordCounter():

    print("WordCounter")
    # 读入数据集
    print("读入数据集")
    filename = "/home/hechaoqun/train_data/data/word2vec/train_corpus.txt"   # 当前目录下
    fr = open(filename, 'r', encoding='UTF-8')
    print("读取成功")
    WordLine = fr.readline()
    while WordLine:
        print("读取数据集")
        for item in WordLine.strip().split():
            words_corpus.append(item)   #语料库
        WordLine = fr.readline()
    
    #words_diff_bef = []
    print("语料库大小：",len(words_corpus))    #语料库大小
    
    stopwords = []
    #停用词
    print("读取停用词")
    filename_stop = "./english"
    fr_stop = open(filename_stop, 'r', encoding='UTF-8')
    print("读取成功")
    WordLine_stop = fr_stop.readline()
    while WordLine_stop:
        for item in WordLine_stop.strip().split():
            stopwords.append(item)
        WordLine_stop = fr_stop.readline()
    print("停用词数量:", len(stopwords))
    # 去重,以及最小词频处理
    print("初级词汇表")
    '''
    for item in words_corpus:
        if item not in words_diff_bef:
            words_diff_bef.append(item)
            word_times[item] = 1
            print(len(words_diff_bef))
        else:
            word_times[item] = word_times[item] + 1
    '''
    words_diff_bef = list(set(words_corpus))

    print("初级词汇表大小:", len(words_diff_bef))
    
    print("词汇表去低词频")
    word_times_bef = Counter(words_corpus)
    for key in word_times_bef.keys():
        if word_times_bef[key] > min_count and key not in stopwords:
            words_diff.append(key)   # 词汇表
            word_times[key] = word_times_bef[key]
            print(len(words_diff))
           # 记录词与其对应的词频
    print("最终词汇表大小:", len(words_diff))

    print("语料库大小：",len(words_corpus))    #语料库大小
    print("词汇表大小：",len(words_diff))   # 词汇表
    #获取词频(并没有去掉频率过低的单词，例如为1)

    pass

#词向量初始化，辅助向量初始化
def initwords():
    print("initwords")
    for item in words_diff:
        word_vec[item] = np.random.random((1, m_length))[0]/m_length
        #代表生成 1 行 m_length 列的浮点数，浮点数都是从0-1中随机。
        # 为什么要在这里加个0呢？
        '''
            那是因为np.random.random() 生成的是一个多维矩阵，即使是一维的，那也是[[1, 2, 3]].
            所以加个[0]，取出第一行的数据
        '''
        word_vec_help[item] = np.zeros(m_length)   # 每一个词向量的辅助向量初始化
    pass

#得到2*window_length 个上下文词
def get_windows(center):
    #print("get_windows")
    res_context = []

    for index in range(1, len(words_corpus)-1): # 前面的词
        i = center - index
        if i >= 0:
            if words_corpus[i] in words_diff:
                res_context.append(words_corpus[i])

        if len(res_context) == 2 * window_length:
            break

        j = center + index
        if j <= len(words_corpus) - 1:
            if words_corpus[j] in words_diff:
                res_context.append(words_corpus[j])

        if len(res_context) == 2 * window_length:
            break

    return res_context   # 返回的是word 列表

#对负采样的词汇表进行带权初始化
def initweight():
    print("initweight")
    sum = 0
    for item in words_diff:
        sum = sum + math.pow(word_times[item], 0.75)   # 计算词频之和

    sum2 = 0
    # 计算每个Lk大小，方便计算区间大小
    for item in words_diff:
        per_distance[item] = math.pow(word_times[item], 0.75) / sum   # 计算每一个词的len(w)
        sum2 = sum2 + per_distance[item]   # 计算 Lk
        count_word_len[item] = sum2

    for item in words_diff:   # 计算区间长度
        distance_word_len[item] = count_word_len[item] / sum2

    pass

#选择负样本词
def choice_NEG(target_word):
    #print("choice_NEG")

    list_NEG = []
    while len(list_NEG) < 5:
        m = random.randrange(1, NEG_M)
        for item in words_diff:
            if m / NEG_M < distance_word_len[item]:
                if item != target_word:
                    list_NEG.append(item)
                    break

    list_NEG.append(target_word)

    return list_NEG

def sigmoid(inx):
    #print("sigmoid")
    if inx >= 0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0 /(1 + np.exp(-inx))
    else:
        return np.exp(inx)/(1 + np.exp(inx))

def CB_NS():
    print("CB_NS")
    initweight()
    global sum_index
    sum_index = 1
    for index in range(len(words_corpus)):   # 开始选词，找窗口

        # 只有词频超过这个阀值的词才能被训练
        if words_corpus[index] in words_diff:
            print("窗口", index, sum_index)
            sum_index = sum_index + 1

            context = get_windows(index)   # 找窗口

            word_center = words_corpus[index]   # 中心词
            NEG_words = choice_NEG(word_center)   # 负样本词和中心词的集合

            global neule   # e
            neule = np.zeros(m_length)

            Xw = np.zeros(m_length)

            for item1 in context:
                Xw = Xw + word_vec[item1]   # 词向量之和

            for item in NEG_words:

                x_vec = np.dot(Xw, word_vec_help[item])
                q = sigmoid(x_vec)

                if item == word_center:
                    Lw = 1
                else:
                    Lw = 0
                g = p_learning * (Lw - q)

                neule = neule + g * word_vec_help[item]

                word_vec_help[item] = word_vec_help[item] + g * Xw

            for item2 in context:

                word_vec[item2] = word_vec[item2] + neule

    pass

if __name__ == '__main__':
    start = time.time()
    print("程序正在执行......")

    WordCounter()   # 对数据集进行处理
    initwords()  # 对词向量初始化
    CB_NS()
    '''
    for index in range(len(words_diff)):
        item = words_diff[index]
        print(index, item, word_times[item], word_vec[item])   #验证
    '''

    end = time.time()
    print("程序运行时间:" + str(end - start) + "s")
