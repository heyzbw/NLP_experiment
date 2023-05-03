import csv

# 构建二元组频率词典
bigram_freq = {}

# 打开语料库文件，逐行读取
with open(r'train.csv/train.csv', "r", encoding="utf-8") as f:
    for line in f:
        # 分词
        words = line.strip().split()
        # 遍历词列表，构建二元组并更新频次
        for i in range(len(words) - 1):
            bigram = (words[i], words[i+1])
            if bigram in bigram_freq:
                bigram_freq[bigram] += 1
            else:
                bigram_freq[bigram] = 1
# 输出二元组频率字典
with open('bi_freq.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for bigram, freq in bigram_freq.items():
        writer.writerow([bigram[0], bigram[1], freq])
