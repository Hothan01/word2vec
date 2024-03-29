//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

// 构建的全局变量

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

// 定义的浮点数
typedef float real;                    // Precision of float numbers

// 词的结构体
struct vocab_word {
	long long cn; // 出现的次数
	int *point; // 从根结点到叶子节点的路径
	char *word, *code, codelen;// 分别对应着词，Huffman编码，编码长度
};

char train_file[MAX_STRING], output_file[MAX_STRING];// 训练文件，输出文件
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab; // 出现的词的统计

// 初始化参数
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int *vocab_hash;// 存储词的hash
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

// 生成负采样的概率表
void InitUnigramTable() {
	int a, i;
	double train_words_pow = 0;
	double d1, power = 0.75;
	table = (int *)malloc(table_size * sizeof(int));// int --> int
	for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power); // 所有词的出现次数的0.75次幂的和
	// 类似轮盘赌生成每个词的概率
	i = 0;
	d1 = pow(vocab[i].cn, power) / train_words_pow;
	for (a = 0; a < table_size; a++) {
		table[a] = i;
		if (a / (double)table_size > d1) {
			i++;
			d1 += pow(vocab[i].cn, power) / train_words_pow;
		}
		if (i >= vocab_size) i = vocab_size - 1; // 考虑最后一个词的频率很大的特例
	}
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
// 读取一个词
void ReadWord(char *word, FILE *fin) {
	int a = 0, ch; // a 用来表示当前词的字符数
	while (!feof(fin)) {
		ch = fgetc(fin);
		if (ch == 13) continue; // 回车，\r
		if ((ch == ' ') || (ch == '\t') || (ch == '\n')) { // 这是啥意思？
			if (a > 0) {// 当前的词还没结束
				if (ch == '\n') ungetc(ch, fin);
				break;
			}
			if (ch == '\n') {
				strcpy(word, (char *)"</s>");// 换行符用</s>表示
				return;
			} else continue;
		}
		word[a] = ch;
		a++;
		if (a >= MAX_STRING - 1) a--;   // Truncate too long words
	}
	word[a] = 0; // 字符串终止符
}

// Returns hash value of a word
// 取词的hash值
int GetWordHash(char *word) {
	unsigned long long a, hash = 0;
	for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
	hash = hash % vocab_hash_size;
	return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
// 查找词在词库中的位置（索引值），若没有查找到则返回-1
int SearchVocab(char *word) {
	unsigned int hash = GetWordHash(word);
	while (1) {
		if (vocab_hash[hash] == -1) return -1;// 不存在该词
		if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];// 返回索引值
		hash = (hash + 1) % vocab_hash_size;
	}
	return -1;// 不存在该词
}

// Reads a word and returns its index in the vocabulary
// 返回的是在词库中的位置
int ReadWordIndex(FILE *fin) {
	char word[MAX_STRING];
	ReadWord(word, fin);
	if (feof(fin)) return -1;
	return SearchVocab(word);
}

// Adds a word to the vocabulary
// 为词库中增加一个词
int AddWordToVocab(char *word) {
	unsigned int hash, length = strlen(word) + 1;// 单词的长度+1
	if (length > MAX_STRING) length = MAX_STRING;
	vocab[vocab_size].word = (char *)calloc(length, sizeof(char));//开始的位置增加指定的词
	strcpy(vocab[vocab_size].word, word);
	vocab[vocab_size].cn = 0; //出现次数
	vocab_size++;
	// Reallocate memory if needed
	if (vocab_size + 2 >= vocab_max_size) {
		vocab_max_size += 1000;
		vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
	}
	hash = GetWordHash(word);// 对增加的词hash
	while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;// hash的碰撞检测
	vocab_hash[hash] = vocab_size - 1;// 词的hash值->词的词库中的索引
	return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
	return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
// 根据词出现的频率对词库中的词排序
void SortVocab() {
	int a, size;
	unsigned int hash;
	// Sort the vocabulary and keep </s> at the first position
	qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
	// 排完序后需要重新做hash运算
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	size = vocab_size;
	train_words = 0;
	for (a = 0; a < size; a++) {
		// Words occuring less than min_count times will be discarded from the vocab
		// 根据min_count对低频词的处理
		if ((vocab[a].cn < min_count) && (a != 0)) {
			vocab_size--;
			free(vocab[a].word); // 这个为什么只释放了word？
		} else {
			// Hash will be re-computed, as after the sorting it is not actual
			hash = GetWordHash(vocab[a].word);
			while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
			vocab_hash[hash] = a;
			train_words += vocab[a].cn;
		}
	}
	vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
	// Allocate memory for the binary tree construction
	// 为构建huffman树申请空间
	for (a = 0; a < vocab_size; a++) {
		vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
		vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	}
}

// Reduces the vocabulary by removing infrequent tokens
// 删除频率较小的词
void ReduceVocab() {
	int a, b = 0;
	unsigned int hash;
	// 通过min_reduce控制
	for (a = 0; a < vocab_size; a++)
		if (vocab[a].cn > min_reduce) {
			vocab[b].cn = vocab[a].cn;
			vocab[b].word = vocab[a].word;
			b++;
		} else free(vocab[a].word);
	vocab_size = b;// 删减后词的个数
	// 重新进行hash操作
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	for (a = 0; a < vocab_size; a++) {
		// Hash will be re-computed, as it is not actual
		hash = GetWordHash(vocab[a].word);
		while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
		vocab_hash[hash] = a;
	}
	fflush(stdout);
	min_reduce++; //这是为了提高降低低频词的门槛，但是为什么调用ReduceVocab()函数的地方不循环判断一下是否已经删够了
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
// 根据词库中的词频构建Huffman树
void CreateBinaryTree() {
	long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
	char code[MAX_CODE_LENGTH];

	// 申请2倍的词的空间，（在这里完全没有必要申请这么多的空间）
	long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));

	// 分成两半进行初始化
	for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;// 前半部分初始化为每个词出现的次数
	for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;// 后半部分初始化为一个固定的较大的常数

	// 两个指针：
	// pos1指向前半截的尾部
	// pos2指向后半截的开始
	pos1 = vocab_size - 1;
	pos2 = vocab_size;

	// Following algorithm constructs the Huffman tree by adding one node at a time
	// 每次增加一个节点，构建Huffman树
	for (a = 0; a < vocab_size - 1; a++) {
		// First, find two smallest nodes 'min1, min2'
		// 选择最小的节点min1
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min1i = pos1;
				pos1--;
			} else {
				min1i = pos2;
				pos2++;
			}
		} else {
			min1i = pos2;
			pos2++;
		}
		// 选择最小的节点min2
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min2i = pos1;
				pos1--;
			} else {
				min2i = pos2;
				pos2++;
			}
		} else {
			min2i = pos2;
			pos2++;
		}

		count[vocab_size + a] = count[min1i] + count[min2i];
		// 设置父节点
		parent_node[min1i] = vocab_size + a;
		parent_node[min2i] = vocab_size + a;
		binary[min2i] = 1;// 设置一个子树的编码为1，另一个子树在初始化时已经为0了
	}
	// Now assign binary code to each vocabulary word
	// 为每一个词分配二进制编码，即Huffman编码
	for (a = 0; a < vocab_size; a++) {// 针对每一个词
		b = a;
		i = 0;
		while (1) {
			code[i] = binary[b];// 找到当前的节点的编码
			point[i] = b;// 记录从叶子节点到根结点的序列
			i++;
			b = parent_node[b];// 找到当前节点的父节点
			if (b == vocab_size * 2 - 2) break;// 已经找到了根结点，根节点是没有编码的； 哈夫曼树一定有N-1个非叶子节点，因此根节点一定在N*2-2
		}
		vocab[a].codelen = i;// 词的编码长度
		vocab[a].point[0] = vocab_size - 2;// 根结点
		for (b = 0; b < i; b++) {
			vocab[a].code[i - b - 1] = code[b];// 编码的反转
			vocab[a].point[i - b] = point[b] - vocab_size;// 记录的是从根结点到叶子节点的路径; 这里为什么要减去vocab_size呢，仅仅是给叶子节点编了一个码？
		}
	}
	free(count);
	free(binary);
	free(parent_node);
}

// 读取输入的文件，并从输入文件中构建词库
void LearnVocabFromTrainFile() {
	char word[MAX_STRING];// 存储每一个单词
	FILE *fin;
	long long a, i;

	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1; // 初始化

	fin = fopen(train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	vocab_size = 0;// 记录文件中的词的个数

	AddWordToVocab((char *)"</s>");// 在最开始增加指定的词

	// 开始从文本取每一个词
	while (1) {
		ReadWord(word, fin); // 读取每一个词
		if (feof(fin)) break; // 判断文件是否读完
		train_words++; // 记录词的个数
		if ((debug_mode > 1) && (train_words % 100000 == 0)) { // 输出已经训练的词数
			printf("%lldK%c", train_words / 1000, 13);
			fflush(stdout);
		}
		i = SearchVocab(word);// 查找词在词库中的位置
		if (i == -1) {// 没有查找到对应的词
			a = AddWordToVocab(word);// 增加词
			vocab[a].cn = 1;// 设置词出现的次数为1
		} else vocab[i].cn++;// 设置词出现的次数+1

		// 根据当前词的个数和设定的hash表的大小，删除低频词
		if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
	}
	SortVocab();// 根据词出现的频率对词进行排序
	if (debug_mode > 0) {
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	file_size = ftell(fin);
	fclose(fin);
}

// 保存词库
void SaveVocab() {
	long long i;
	FILE *fo = fopen(save_vocab_file, "wb");
	// 保存词库时，保存的是词库中的词和词出现的次数
	for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
	fclose(fo);
}

void ReadVocab() {
	long long a, i = 0;
	char c;
	char word[MAX_STRING];
	FILE *fin = fopen(read_vocab_file, "rb");
	if (fin == NULL) {
		printf("Vocabulary file not found\n");
		exit(1);
	}
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1; // 初始化vocab_hash
	vocab_size = 0;
	while (1) {
		ReadWord(word, fin);
		if (feof(fin)) break;
		a = AddWordToVocab(word);
		fscanf(fin, "%lld%c", &vocab[a].cn, &c); // 从文件中读取词的出现次数
		i++;
	}
	SortVocab();
	if (debug_mode > 0) { //
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	fin = fopen(train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	fseek(fin, 0, SEEK_END);
	file_size = ftell(fin); //得到训练文件的字节数
	fclose(fin);
}

// 初始化网络
// 主要分为两个部分：1、对词向量的初始化；2、对映射层到输出层权重的初始化
void InitNet() {
	long long a, b;
	unsigned long long next_random = 1;

	// 为每一个词分配词向量的空间
	// 对齐分配内存,posix_memalign函数的用法类似于malloc的用法，最后一个参数的分配的内存的大小
	a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real)); // layer1_size为词向量的长度
	if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}

	// 层次softmax的结构
	if (hs) { //采用hierarchy softmax
		// 映射层到输出层之间的权重
		// 分配huffman树中的非叶子节点对应的向量的空间
		a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
		if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
		for (a = 0; a < vocab_size; a++)
			for (b = 0; b < layer1_size; b++)
				syn1[a * layer1_size + b] = 0;// 权重初始化为0
	}

	// 负采样的结构
	if (negative > 0) {
		a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
		if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
		for (a = 0; a < vocab_size; a++)
			for (b = 0; b < layer1_size; b++)
				syn1neg[a * layer1_size + b] = 0;
	}

	// 词向量随机初始化
	for (a = 0; a < vocab_size; a++)
		for (b = 0; b < layer1_size; b++) {
			next_random = next_random * (unsigned long long)25214903917 + 11;
			// 1、与：相当于将数控制在一定范围内
			// 2、0xFFFF：65536
			// 3、/65536：[0,1]之间
			syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;// 初始化词向量
		}

	// 构建Huffman树
	CreateBinaryTree();
}

void *TrainModelThread(void *id) {
	long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
	long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
	long long l1, l2, c, target, label, local_iter = iter;
	unsigned long long next_random = (long long)id;
	real f, g;
	clock_t now;

	// layer1_size为词向量的长度
	real *neu1 = (real *)calloc(layer1_size, sizeof(real));// 存储映射层的结果(CBOW方法中中各个周围词向量的和)
	real *neu1e = (real *)calloc(layer1_size, sizeof(real));// 词向量更新量

	FILE *fi = fopen(train_file, "rb");
	// 利用多线程对训练文件划分，每个线程训练一部分的数据
	fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);

	// 训练模型的核心部分
	while (1) {
		// 每处理10000个词重新计算学习率
		if (word_count - last_word_count > 10000) {// 每处理10000个词重新计算学习率
			word_count_actual += word_count - last_word_count; //word_count_actual是各个线程处理的词的和
			last_word_count = word_count;
			if ((debug_mode > 1)) {
				now = clock();
				printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
				       word_count_actual / (real)(iter * train_words + 1) * 100,
				       word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
				fflush(stdout);
			}
			// 重新计算alpha的值
			alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
			// 防止学习率过小
			if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
		}

		// sentence_length=0表示的是当前还没有读取文本
		// 开始读取文本，读取词的个数最多为MAX_SENTENCE_LENGTH
		if (sentence_length == 0) {
			// 需要根据文件指针的位置读取相应的文本
			while (1) {
				word = ReadWordIndex(fi);// 词在词库中的索引（注意是数字）

				if (feof(fi)) break;
				if (word == -1) continue;// 没有查到该词；相当于把训练语料中不在词表中的词都删掉
				word_count++;

				if (word == 0) break;//读到的是</S>——换行符，表示一个句子读完了

				// The subsampling randomly discards frequent words while keeping the ranking same
				if (sample > 0) {
					real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn; //这是什么采样原理？
					next_random = next_random * (unsigned long long)25214903917 + 11;
					if (ran < (next_random & 0xFFFF) / (real)65536) continue;
				}

				sen[sentence_length] = word;// 存储词在词库中的位置，word代表的是Index
				sentence_length++;
				if (sentence_length >= MAX_SENTENCE_LENGTH) break;// 达到指定长度
			}
			sentence_position = 0;// 将待处理的文本指针置0
		}

		// 当前的线程已经处理完分配给该线程的文本
		if (feof(fi) || (word_count > train_words / num_threads)) {// 当前线程已经读完数据
			word_count_actual += word_count - last_word_count;
			// 当前线程的迭代次数
			local_iter--;
			if (local_iter == 0) break;// 所有迭代结束，随后该线程结束
			// 重新置0，准备下一次重新迭代
			word_count = 0;
			last_word_count = 0;
			sentence_length = 0;
			// 重置文件指针
			fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
			continue;
		}

		// sen表示的是当前的线程读取到的每一个词对应在词库中的索引
		word = sen[sentence_position];//sentence_position表示的是当前词
		if (word == -1) continue;

		// 初始化映射层
		for (c = 0; c < layer1_size; c++) neu1[c] = 0;// 映射层的结果
		for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

		// 产生一个0~window-1的随机数
		next_random = next_random * (unsigned long long)25214903917 + 11;
		b = next_random % window;

		// 模型的训练
		if (cbow) {  // 训练CBOW模型
			// input -> hidden  输入层到映射层
			cw = 0; //周围词的数目
			for (a = b; a < window * 2 + 1 - b; a++) //为什么要用随机窗口大小？
				if (a != window) {
					c = sentence_position - window + a;// sentence_position表示的是当前词的位置；窗口范围：[sentense_position-window+b,sentense_position+window-b]
					// 判断c是否越界
					if (c < 0) continue;
					if (c >= sentence_length) continue;

					last_word = sen[c];// 找到c对应的索引
					if (last_word == -1) continue; //之前已经判断过不要把不在词库的词加入

					for (c = 0; c < layer1_size; c++)
						neu1[c] += syn0[c + last_word * layer1_size];// 把各个周围词向量累加
					cw++;
				}

			if (cw) { //周围词数大于0，开始计算梯度并更新
				for (c = 0; c < layer1_size; c++)
					neu1[c] /= cw;// 计算均值

				// 计算的中心词是word
				// 层次Softmax
				if (hs)
					for (d = 0; d < vocab[word].codelen; d++) {// word为当前词
						// 计算输出层的输出
						f = 0;
						l2 = vocab[word].point[d] * layer1_size;// 找到当前词的huffman路径中的第d个节点，以便下面找到这个节点的向量存储位置
						// Propagate hidden -> output
						for (c = 0; c < layer1_size; c++)
							f += neu1[c] * syn1[c + l2];// 周围词向量的平均值与当前词huffman树路径上当前节点向量的点积

						if (f <= -MAX_EXP) continue;
						else if (f >= MAX_EXP) continue;
						else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];// Sigmoid结果

						// 'g' is the gradient multiplied by the learning rate
						g = (1 - vocab[word].code[d] - f) * alpha;
						// Propagate errors output -> hidden
						for (c = 0; c < layer1_size; c++)
							neu1e[c] += g * syn1[c + l2];// 修改词向量更新量
						// Learn weights hidden -> output
						for (c = 0; c < layer1_size; c++)
							syn1[c + l2] += g * neu1[c];// 修改huffman树路径上节点的向量
					}
				// NEGATIVE SAMPLING  采用负采样
				if (negative > 0)
					for (d = 0; d < negative + 1; d++) {
						// 标记target和label
						if (d == 0) {// 正样本
							target = word;
							label = 1;
						} else {// 选择出负样本
							next_random = next_random * (unsigned long long)25214903917 + 11;
							target = table[(next_random >> 16) % table_size];// 从table表中选择出负样本
							// 重新选择
							if (target == 0) target = next_random % (vocab_size - 1) + 1; // 选出</S>，则重新选
							if (target == word) continue; //选出正样本，则重新选
							label = 0;
						}

						l2 = target * layer1_size;
						f = 0;
						for (c = 0; c < layer1_size; c++)
							f += neu1[c] * syn1neg[c + l2];// 映射层到输出层；求周围词向量平均值和参数向量θ的点积

						// g
						if (f > MAX_EXP) g = (label - 1) * alpha;	// sigmoid函数太大认为是1
						else if (f < -MAX_EXP) g = (label - 0) * alpha; 	//sigmoid函数太小认为是0
						else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

						for (c = 0; c < layer1_size; c++)
							neu1e[c] += g * syn1neg[c + l2]; // 更新e
						for (c = 0; c < layer1_size; c++)
							syn1neg[c + l2] += g * neu1[c]; // 更新θ
					}
				// hidden -> in
				// 以上是从映射层到输出层的修改，现在返回修改每一个词向量
				for (a = b; a < window * 2 + 1 - b; a++)  // 取窗口词函数，同上（遍历每一个周围词）
					if (a != window) {
						c = sentence_position - window + a;
						if (c < 0) continue;
						if (c >= sentence_length) continue;
						last_word = sen[c];
						if (last_word == -1) continue;
						// 利用窗口内的所有词向量的梯度之和来更新
						for (c = 0; c < layer1_size; c++)
							syn0[c + last_word * layer1_size] += neu1e[c];
					}
			}
		} else {  //train skip-gram 训练skip-gram模型
			for (a = b; a < window * 2 + 1 - b; a++) // 遍历周围词
				if (a != window) {
					c = sentence_position - window + a; // sentence_position表示的是当前词的位置；窗口范围：[sentense_position-window+b,sentense_position+window-b]
					if (c < 0) continue;
					if (c >= sentence_length) continue;
					last_word = sen[c]; // 当前周围词

					if (last_word == -1) continue;
					l1 = last_word * layer1_size; // 当前周围词的词向量存储偏置

					for (c = 0; c < layer1_size; c++) //词向量更新量初始化
						neu1e[c] = 0;
					// HIERARCHICAL SOFTMAX
					if (hs)
						for (d = 0; d < vocab[word].codelen; d++) {
							f = 0;
							l2 = vocab[word].point[d] * layer1_size; // huffman树的非叶子节点的向量偏置
							// Propagate hidden -> output
							// 映射层即为输入层
							for (c = 0; c < layer1_size; c++) // 计算词向量和参数的点积
								f += syn0[c + l1] * syn1[c + l2];

							if (f <= -MAX_EXP) continue; //计算sigmoid
							else if (f >= MAX_EXP) continue;
							else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

							// 'g' is the gradient multiplied by the learning rate
							g = (1 - vocab[word].code[d] - f) * alpha;
							// Propagate errors output -> hidden
							for (c = 0; c < layer1_size; c++)
								neu1e[c] += g * syn1[c + l2]; //更新词向量更新量e
							// Learn weights hidden -> output
							for (c = 0; c < layer1_size; c++)
								syn1[c + l2] += g * syn0[c + l1]; // 更新非叶子节点参数
						}
					// NEGATIVE SAMPLING
					if (negative > 0)
						for (d = 0; d < negative + 1; d++) { // 负采样
							if (d == 0) { // 正样本
								target = word;
								label = 1;
							} else { // 负样本（改变的同样是中心词）
								next_random = next_random * (unsigned long long)25214903917 + 11;
								target = table[(next_random >> 16) % table_size];
								if (target == 0) target = next_random % (vocab_size - 1) + 1;
								if (target == word) continue;
								label = 0;
							}
							l2 = target * layer1_size; // 负采样词（中心词）的偏置
							f = 0;
							for (c = 0; c < layer1_size; c++)  // 当前周围词向量和负采样词参数的点积
								f += syn0[c + l1] * syn1neg[c + l2];

							if (f > MAX_EXP) g = (label - 1) * alpha; // 计算sigmoid函数
							else if (f < -MAX_EXP) g = (label - 0) * alpha;
							else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

							for (c = 0; c < layer1_size; c++)  // 词向量更新量
								neu1e[c] += g * syn1neg[c + l2];

							for (c = 0; c < layer1_size; c++) // 参数更新
								syn1neg[c + l2] += g * syn0[c + l1];
						}

					// Learn weights input -> hidden
					for (c = 0; c < layer1_size; c++)
						syn0[c + l1] += neu1e[c]; // 更新词向量
				}
		}
		// 当已经处理完读入的所有文本，要重新继续往下读文本
		sentence_position++;
		if (sentence_position >= sentence_length) {
			sentence_length = 0;
			continue;
		}
	}
	fclose(fi);
	free(neu1);
	free(neu1e);
	pthread_exit(NULL);
}

// 模型训练
void TrainModel() {
	long a, b, c, d;
	FILE *fo;

	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));// 多线程id数组

	printf("Starting training using file %s\n", train_file);
	starting_alpha = alpha;

	// 区分是否指定词库
	// 若指定词库，则从词库中读入词
	// 若不指定词库，则从文件中构建词库
	if (read_vocab_file[0] != 0)
		ReadVocab();// 指定词库
	else
		LearnVocabFromTrainFile();// 不指定词库，从文件中构建词库

	if (save_vocab_file[0] != 0) SaveVocab();// 判断是否需要保存词库

	// 若没有指定输出文件，则退出
	if (output_file[0] == 0) return;

	InitNet();// 初始化网络

	if (negative > 0) InitUnigramTable();// 利用负采样的方法

	// 开始训练
	start = clock();
	for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
	for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL); // 等待所有线程结束

	// 输出最终的训练结果
	fo = fopen(output_file, "wb");
	if (classes == 0) {  // Save the word vectors
		fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
		for (a = 0; a < vocab_size; a++) {
			fprintf(fo, "%s ", vocab[a].word);
			if (binary) // 二进制文件
				for (b = 0; b < layer1_size; b++)
					fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
			else
				for (b = 0; b < layer1_size; b++)
					fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
			fprintf(fo, "\n");
		}
	} else { // 输出词类而非词向量
		// Run K-means on the word vectors
		int clcn = classes, iter = 10, closeid;
		int *centcn = (int *)malloc(classes * sizeof(int));
		int *cl = (int *)calloc(vocab_size, sizeof(int));
		real closev, x;
		real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
		for (a = 0; a < vocab_size; a++)
			cl[a] = a % clcn;
		for (a = 0; a < iter; a++) {
			for (b = 0; b < clcn * layer1_size; b++)
				cent[b] = 0;
			for (b = 0; b < clcn; b++)
				centcn[b] = 1;
			for (c = 0; c < vocab_size; c++) {
				for (d = 0; d < layer1_size; d++)
					cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
				centcn[cl[c]]++;
			}
			for (b = 0; b < clcn; b++) {
				closev = 0;
				for (c = 0; c < layer1_size; c++) {
					cent[layer1_size * b + c] /= centcn[b];
					closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
				}
				closev = sqrt(closev);
				for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
			}
			for (c = 0; c < vocab_size; c++) {
				closev = -10;
				closeid = 0;
				for (d = 0; d < clcn; d++) {
					x = 0;
					for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
					if (x > closev) {
						closev = x;
						closeid = d;
					}
				}
				cl[c] = closeid;
			}
		}
		// Save the K-means classes
		for (a = 0; a < vocab_size; a++)
			fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
		free(centcn);
		free(cent);
		free(cl);
	}
	fclose(fo);
}

// 解析命令行
int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++)
		if (!strcmp(str, argv[a])) {// 查找对应的参数
			if (a == argc - 1) {
				printf("Argument missing for %s\n", str);
				exit(1);
			}
			return a;// 匹配成功，返回值所在的位置
		}
	return -1;
}

int main(int argc, char **argv) {
	int i;
	//  判断参数的个数
	if (argc == 1) {
		printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-train <file>\n");
		printf("\t\tUse text data from <file> to train the model\n");
		printf("\t-output <file>\n");
		printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
		printf("\t-size <int>\n");
		printf("\t\tSet size of word vectors; default is 100\n");
		printf("\t-window <int>\n");
		printf("\t\tSet max skip length between words; default is 5\n");
		printf("\t-sample <float>\n");
		printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
		printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
		printf("\t-hs <int>\n");
		printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
		printf("\t-negative <int>\n");
		printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads (default 12)\n");
		printf("\t-iter <int>\n");
		printf("\t\tRun more training iterations (default 5)\n");
		printf("\t-min-count <int>\n");
		printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
		printf("\t-alpha <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
		printf("\t-classes <int>\n");
		printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
		printf("\t-debug <int>\n");
		printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
		printf("\t-binary <int>\n");
		printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
		printf("\t-save-vocab <file>\n");
		printf("\t\tThe vocabulary will be saved to <file>\n");
		printf("\t-read-vocab <file>\n");
		printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
		printf("\t-cbow <int>\n");
		printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
		printf("\nExamples:\n");
		printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
		return 0;
	}

	output_file[0] = 0;// 输出文件
	save_vocab_file[0] = 0;// 输出词的文件
	read_vocab_file[0] = 0;// 读入指定词的文件

	// 解析word2vec所需用到的参数
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
	if (cbow) alpha = 0.05;
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);

	vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));// 存储每一个词的结构体
	vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));// 存储词的hash
	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));// 申请EXP_TABLE_SIZE+1个空间

	// 计算sigmoid值
	for (i = 0; i < EXP_TABLE_SIZE; i++) {
		expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
	}

	// 开始模型训练
	TrainModel();// 模型训练
	return 0;
}