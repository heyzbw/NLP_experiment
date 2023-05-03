import pandas as pd

# 从train.csv获取词典（无频数）
vocabulary = set()
data = pd.read_csv(r'train.csv/train.csv', header='infer')
for i in range(data.sentence.shape[0]):
    for j in range(len(data.sentence[i].split())):
        vocabulary.add(data.sentence[i].split()[j])
vo_list = list(vocabulary)
vo_df = pd.DataFrame(vo_list, columns=['words'])
vo_df.to_csv("getvo_all.csv", index=False, header=None)
