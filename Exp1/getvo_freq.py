import csv
from collections import Counter

corpus = Counter()

with open(r'train.csv/train.csv', "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=" ")
    for row in reader:
        corpus.update(row)
with open('vo_freq.csv', mode='w', newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    # 逐行写入数据
    for row in corpus.most_common():
        writer.writerow(row)
