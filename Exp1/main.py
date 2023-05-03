import pandas as pd

# 最初实现的最大匹配算法（未使用频数）
MAXLEN = 14
vocabulary = set()
# 自制词库
with open('getvo_freq.csv', encoding='utf-8') as vo_a:
    for l in vo_a.readlines():
        vocabulary.add(l.split(',')[0])
with open('get_token.csv', encoding='utf-8') as vo_n:
    for l in vo_n.readlines():
        vocabulary.add(l.split(',')[0])


# 正向最大匹配分词
def forw_split(text):
    spt_words = []
    while text != '':
        sub_strg = text[:MAXLEN]
        while sub_strg != '':
            if sub_strg in vocabulary or len(sub_strg) == 1:
                spt_words.append(sub_strg)
                break
            else:
                sub_strg = sub_strg[:-1]
        text = text[len(sub_strg):]
    spt_lst = " ".join(str(i) for i in spt_words)
    return spt_lst

# 逆向最大分词
def backw_split(text):
    spt_words = []
    while text != '':
        sub_strg = text[-MAXLEN:]
        while sub_strg != '':
            if sub_strg in vocabulary or len(sub_strg) == 1:
                spt_words.append(sub_strg)
                break
            else:
                sub_strg = sub_strg[1:]
        text = text[:-len(sub_strg):]
    spt_lst = " ".join(str(i) for i in spt_words[::-1])
    return spt_lst


# 比较正向逆向词汇数以及颗粒粗细最优分词结果
def compare(forw_lst, backw_lst):
    if len(forw_lst.split()) < len(backw_lst.split()):
        return forw_lst
    # elif len(forw_lst.split()) > len(backw_lst.split()):
    #     return backw_lst
    else:
        forw_get = [(1 if w in vocabulary else 0) for w in forw_lst]
        backw_get = [(1 if w in vocabulary else 0) for w in backw_lst]
        forw_rate = sum(forw_get) / len(forw_lst)
        backw_rate = sum(backw_get) / len(backw_lst)
        if forw_rate > backw_rate:
            return forw_lst
        elif backw_rate > forw_rate:
            return backw_lst
        else:
            return forw_lst


# 切分结果转换脚本
def transfer(raw_sen):
    count = 0
    tmp_list = []
    for ele in raw_sen.strip().split(' '):
        _tmp_list = []
        for _ in range(len(ele)):
            _tmp_list.append(count)
            count += 1
        tmp_list.append(str(_tmp_list).replace(' ', ''))

    return ' '.join(tmp_list)


if __name__ == '__main__':
    data = pd.read_csv(r'test.csv', header='infer')
    for i in range(data.sentence.shape[0]):
        text_string = data.sentence[i]
        forw_lst = forw_split(text_string)
        backw_lst = backw_split(text_string)
        better_lst = compare(forw_lst, backw_lst)
        result_lst = transfer(better_lst)
        data.loc[i, 'sentence'] = result_lst
    data.rename(columns={'sentence': 'expected'}, inplace=True)
    data.to_csv(r'main.csv', index=False)
