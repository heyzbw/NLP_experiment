import math
import csv
import pandas as pd

# 使用互信息与二元组频率字典改进的最大匹配算法


# 计算两个词的互信息
def mutual_info(bi_freq, word_freq, word1, word2):
    num_word = sum(int(value) for value in word_freq.values())
    if word1 + word2 in word_freq and word1 in word_freq and word2 in word_freq:
        freq_word1 = int(word_freq[word1])
        freq_word2 = int(word_freq[word2])
        freq_bi = int(word_freq[word1 + word2])
        return math.log((freq_bi * num_word) / (freq_word1 * freq_word2), 2)
    elif (word1, word2) in bi_freq and word1 in word_freq and word2 in word_freq:
        freq_word1 = int(word_freq[word1])
        freq_word2 = int(word_freq[word2])
        freq_bi = int(bi_freq[(word1, word2)])
        return math.log((freq_bi * num_word) / (freq_word1 * freq_word2), 2)
    else:
        return 0.0


# 正向最大匹配分词
def forw_split(bi_freq, word_freq, text):
    spt_words = []
    MAXLEN = 15
    flag = False
    while text != '':
        if len(text) < MAXLEN:
            MAXLEN = len(text)
        sub_strg = text[:MAXLEN]
        while sub_strg != '':
            if sub_strg in word_freq or len(sub_strg) == 1:
                sub_strg_2 = sub_strg[:-1]
                # 如果在字典中或者长度为1
                if sub_strg_2 in word_freq or len(sub_strg_2) == 1:
                    if len(sub_strg) == len(text):
                        spt_words.append(sub_strg)
                        flag = False
                        break
                    else:
                        term3 = sub_strg[-1]+text[len(sub_strg)]
                    if term3 in word_freq and len(term3) != 1:
                        if sub_strg[-1] not in word_freq or text[len(sub_strg)] not in word_freq:
                            spt_words.append(sub_strg)
                            flag = False
                            break
                        else:
                            MI1 = mutual_info(bi_freq, word_freq, sub_strg_2, sub_strg[-1])
                            MI2 = mutual_info(bi_freq, word_freq, sub_strg[-1], text[len(sub_strg)])
                            if MI1 > MI2 or MI1 == MI2:
                                spt_words.append(sub_strg)
                                flag = False
                                # 处理完sub_strg，跳出循环重新去sub_strg
                                break
                            else:
                                spt_words.append(sub_strg_2)
                                spt_words.append(term3)
                                flag = True
                                break
                    # term3不在字典中，则直接切分sub_strg
                    else:
                        spt_words.append(sub_strg)
                        flag = False
                        break
                # 如果term2不在字典，则将term1切分出去
                else:
                    spt_words.append(sub_strg)
                    flag=False
                    break
            else:
                sub_strg = sub_strg[:-1]
        # 跳出循环，更新text
        if flag:
            text = text[len(sub_strg) + 1:]
        else:
            text = text[len(sub_strg):]
    spt_lst = " ".join(str(i) for i in spt_words)
    return spt_lst


# 逆向最大匹配分词
def backw_split(bi_freq, word_freq, text):
    spt_words = []
    MAXLEN = 15
    flag = False
    while text != '':
        if len(text) < MAXLEN:
            MAXLEN = len(text)
        sub_strg = text[-MAXLEN:]
        while sub_strg != '':
            if sub_strg in word_freq or len(sub_strg) == 1:
                sub_strg_2 = sub_strg[1:]
                # 如果sub_strg_2在字典中或者他的长度为1
                if sub_strg_2 in word_freq or len(sub_strg_2) == 1:
                    if len(sub_strg) == len(text):
                        spt_words.insert(0, sub_strg)
                        flag = False
                        break
                    else:
                        term3 = text[text.index(sub_strg) - 1] + sub_strg[0]
                    # 如果term3在字典中且不在字符串开头
                    if term3 in word_freq and len(term3) != 1:
                        # 计算互信息
                        if text[text.index(sub_strg) - 1] not in word_freq or sub_strg[0] not in word_freq:
                            spt_words.insert(0, sub_strg)
                            flag = False
                            break
                        else:
                            MI1 = mutual_info(bi_freq, word_freq, sub_strg_2[-1], sub_strg[-1])
                            MI2 = mutual_info(bi_freq, word_freq, text[text.index(sub_strg) - 1], sub_strg[-1])
                            if MI1 > MI2 or MI1 == MI2:
                                spt_words.insert(0, sub_strg)
                                flag = False
                                # 已经处理完sub_strg了，跳出循环重新去sub_strg
                                break
                            else:
                                spt_words.insert(0, term3)
                                spt_words.insert(0, sub_strg_2)
                                flag = True
                                break
                    # term3不在字典中，则直接切分sub_strg
                    else:
                        spt_words.insert(0, sub_strg)
                        flag = False
                        break
                # 如果term2不在字典，则将term1切分出去
                else:
                    spt_words.insert(0, sub_strg)
                    flag = False
                    break
            else:
                sub_strg = sub_strg[1:]
        # 跳出循环，更新text
        if flag:
            text = text[:text.index(sub_strg) - 1]
        else:
            text = text[:len(text) - len(sub_strg)]
    spt_lst = " ".join(str(i) for i in spt_words)
    return spt_lst


# 比较正向逆向词汇数以及颗粒粗细最优分词结果
def bi_split(bi_freq, word_freq, text):
    forw_lst = forw_split(bi_freq, word_freq, text)
    backw_lst = backw_split(bi_freq, word_freq, text)
    if len(forw_lst.split()) < len(backw_lst.split()):
        return forw_lst
    # elif len(forw_lst.split()) > len(backw_lst.split()):
    #     return backw_lst
    else:
        forw_get = [(1 if w in word_freq else 0) for w in forw_lst]
        backw_get = [(1 if w in word_freq else 0) for w in backw_lst]
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
    # 读入分词
    with open('getvo_freq.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        # 跳过CSV文件的第一行
        next(reader)
        word_freq = {rows[0]: rows[1] for rows in reader}
    # 读入人名
    with open('get_token.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        # 跳过CSV文件的第一行
        next(reader)
        word_n = {rows[0]: rows[1] for rows in reader}
    word_freq.update(word_n)
    # 读入二元组频率
    with open('bi_freq.csv', mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        # 跳过CSV文件的第一行
        next(reader)
        bi_freq = {(rows[0], rows[1]): rows[2] for rows in reader}
    # 分词并输出结果
    data = pd.read_csv(r'test.csv', header='infer')
    for i in range(data.sentence.shape[0]):
        text_string = data.sentence[i]
        # forw_lst = forw_split(bi_freq, word_freq, text_string)
        # backw_lst = backw_cut(text_string, bi_freq, word_freq)
        bi_lst = bi_split(bi_freq, word_freq, text_string)
        result_lst = transfer(bi_lst)
        data.loc[i, 'sentence'] = result_lst
    data.rename(columns={'sentence': 'expected'}, inplace=True)
    data.to_csv(r'main_freq.csv', index=False)
