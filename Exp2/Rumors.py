import pandas as pd
import jieba
import jieba.posseg as pseg
import wordcloud
import matplotlib.pyplot as plt
import hanlp

tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)


# 分词并去掉常用连词
def split_word(sentence):
    with open('stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
    # 去除停用词中的换行符
    stopwords = [word.strip() for word in stopwords]
    spt_w = jieba.cut(sentence)
    return [word for word in spt_w if word not in stopwords]


def han_word(sentence):
    with open('stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
    # 去除停用词中的换行符
    stopwords = [word.strip() for word in stopwords]
    spt_w = tok(sentence)
    return [word for word in spt_w if word not in stopwords]


# 取出大小排名前n的item
def get_order(dict, n):
    sorted_values = sorted(dict.values(), reverse=True)[0:n]
    new_dict = {}
    for key, value in dict.items():
        if value in sorted_values:
            new_dict[key] = value
    return new_dict


if __name__ == '__main__':
    df = pd.read_csv(r'DXYRumors.csv', dtype=str)
    df.loc[:, 'spt_word'] = df['title'].apply(han_word)
    df_co = pd.Series(df['spt_word'].sum()).value_counts()
    dict_all = df_co.to_dict()
    # 获取key为名词的item
    dict_nouns = {}
    for key, value in dict_all.items():
        words = pseg.lcut(key)
        if any(word.flag.startswith('n') for word in words):
            dict_nouns[key] = value
    # 获取key为动词的item
    dict_verb = {}
    for key, value in dict_all.items():
        words = pseg.lcut(key)
        if any(word.flag.startswith('v') for word in words):
            dict_verb[key] = value
    dict_all = get_order(dict_all, 10)
    dict_nouns = get_order(dict_nouns, 10)
    dict_verb = get_order(dict_verb, 10)
    # mask_rumors = plt.imread('./mask/rumors.jpg')
    wc = wordcloud.WordCloud(background_color='white', font_path=r'./smiley/SmileySans-Oblique.ttf')
    wc.fit_words(dict_all)
    wc.to_file(r'./cloud/' + '谣言' + '.jpg')
    wc.fit_words(dict_nouns)
    plt.imshow(wc)
    plt.show()
    wc.to_file(r'./cloud/' + '谣言名词' + '.jpg')
    wc.fit_words(dict_verb)
    plt.imshow(wc)
    plt.show()
    wc.to_file(r'./cloud/' + '谣言动词' + '.jpg')
    plt.imshow(wc)
    plt.show()
