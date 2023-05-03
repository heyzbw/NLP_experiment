import pandas as pd
import jieba
import hanlp

tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)


def split_word(sentence):
    spt_w = jieba.cut(sentence)
    spt_lst = " ".join(str(i) for i in spt_w)
    return spt_lst


def han_word(sentence):
    spt_w = tok(sentence)
    spt_lst = " ".join(str(i) for i in spt_w)
    return spt_lst


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
        # jieba_lst = split_word(text_string)
        han_lst = han_word(text_string)
        result_lst = transfer(han_lst)
        data.loc[i, 'sentence'] = result_lst
    data.rename(columns={'sentence': 'expected'}, inplace=True)
    data.to_csv(r'main_tool.csv', index=False)
