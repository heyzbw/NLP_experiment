import pandas as pd
import jieba
import jieba.posseg as pseg
import hanlp

tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)


# 分词
def split_word(sentence):
    spt_w = jieba.cut(sentence)
    return list(spt_w)

def han_word(sentence):
    spt_w = tok(sentence)
    return spt_w


if __name__ == '__main__':
    df = pd.read_csv(r'test.csv', header='infer')
    df.loc[:, 'spt_word'] = df['sentence'].apply(han_word)
    df_co = pd.Series(df['spt_word'].sum()).value_counts()
    dict_all = df_co.to_dict()
    # 获取key为人名的item
    dict_n = {}
    for key, value in dict_all.items():
        words = pseg.lcut(key)
        if any(word.flag.startswith('nr') for word in words):
            dict_n[key] = value
        # if any(word.flag.startswith('nt') for word in words):
        #     dict_n[key] = value
    print(dict_n)
    df_n = pd.DataFrame.from_dict(dict_n, orient='index')
    df_n.to_csv(r'dict_n.csv', index=True)
