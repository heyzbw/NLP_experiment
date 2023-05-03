import math
import csv
import pandas as pd


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
    with open('vo_freq.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        # 跳过CSV文件的第一行
        next(reader)
        word_freq = {rows[0]: rows[1] for rows in reader}
    # 读入人名
    with open('dict_n.csv', 'r', encoding='utf-8') as file:
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
    data.to_csv(r'submission_freq.csv', index=False)


# 基于互信息和二元组频率的正向最大匹配
# def forw_cut(text, bi_freq, word_freq):
#     # 计算总词数
#     total_num = sum(int(value) for value in word_freq.values())
#     # 初始化动态规划数组
#     n = len(text)
#     dp = [0] * n
#     spt_words = []
#     # 动态规划
#     i = 0
#     while i < n:
#         # 从前往后找最长的匹配
#         max_len = min(MAXLEN, n - i + 1)
#         max_word = None
#         for j in range(1, max_len):
#             word = text[i:i + j]
#             if word in word_freq:
#                 # 当前词在词频字典中，作为一个词
#                 cur_score = int(word_freq[word]) / total_num
#             elif len(word) == 1:
#                 # 单字词，直接作为一个词
#                 cur_score = 1e-8 / total_num
#             else:
#                 # 当前词不在词频字典中，使用互信息计算分数
#                 cur_score = max(mutual_info(bi_freq, word_freq, total_num, text[i:i + k], text[i + k:i + j])
#                                 for k in range(j - 1)) + 1e-8
#             # 更新最大分数和最大词
#             if cur_score > dp[i]:
#                 dp[i] = cur_score
#                 max_word = word
#         # 将当前词加入结果列表
#         if max_word is not None:
#             spt_words.append(max_word)
#             # yield max_word
#         # 更新下一次搜索的起始位置
#         if max_word is None:
#             k = 1
#         else:
#             k = len(max_word)
#         i += k
#         spt_lst = " ".join(str(i) for i in spt_words)
#     return spt_lst
#
#
# # 基于互信息和二元组频率的逆向最大匹配
# def backw_cut(text, bi_freq, word_freq):
#     # 计算总词数
#     total_num = sum(int(value) for value in word_freq.values())
#     # 初始化动态规划数组
#     n = len(text)
#     dp = [0] * n
#     spt_words = []
#     # 动态规划
#     i = 0
#     while i < n:
#         # 从后往前找最长的匹配
#         max_len = min(MAXLEN, n - i)
#         max_word = None
#         for j in range(max_len, 0, -1):
#             word = text[i:i + j]
#             if word in word_freq and len(word) != 1:
#                 # 当前词在词频字典中，作为一个词
#                 cur_score = int(word_freq[word]) / total_num
#             elif len(word) == 1:
#                 # 单字词，直接作为一个词
#                 cur_score = 1 / total_num
#             else:
#                 # 当前词不在词频字典中，使用互信息计算分数
#                 cur_score = max(mutual_info(bi_freq, word_freq, total_num, text[i:i + k], text[i + k:i + j])
#                                 for k in range(j - 1)) + 1e-8
#             # 更新最大分数和最大词
#             if cur_score > dp[i]:
#                 dp[i] = cur_score
#                 max_word = word
#         # 将当前词加入结果列表
#         if max_word is not None:
#             spt_words.append(max_word)
#             # yield max_word
#         # 更新下一次搜索的起始位置
#         if max_word is None:
#             k = 1
#         else:
#             k = len(max_word)
#         i += k
#         spt_lst = " ".join(str(i) for i in spt_words)
#     return spt_lst