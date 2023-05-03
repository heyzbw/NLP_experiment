import pandas as pd
import jieba
import jieba.analyse
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


#添加未登录词
def addword():
    entity_list = ['新型冠状病毒/8/nz', '冠状病毒/8/nz', '疫情地图/8/nz', '武汉肺炎/8/nz', '人传人', '病毒/8/nz',
                   '疫情/8/nz',
                   '病例', '新冠/8/nz', 'SARS/8/nz', '隔离', '口罩', '卫健委/5/nt', '双黄连/8/nz', '埃博拉/3/nz',
                   '瑞德西韦/5/nz',
                   '金银潭/5/nz', '百步亭/5/nz', '火神山/5/nz', '雷神山/5/nz', '张家界/5/ns', '小汤山/5/nz',
                   '威斯特丹号/5/nz', '钻石公主号/5/nz',
                   '方舱医院/5/nz', '同济医院/3/nz', '协和医院/3/nz', '紫外线', '蝙蝠/3/nz', '酒精', '消毒', '确诊',
                   '连闯/2/v',
                   '康复/3/n', '封城/3/n', '汤圆/3/n', '武软/3/nt', '红十字会/3/nt', '马拉松/3/n', '小姐姐/3/r',
                   '医护人员/5/nr', '医疗队/5/nr', '医院院长/5/nr', '张定宇/5/nr', '护士/5/nr', '医生/5/nr',
                   '钟南山/5/nr', '李兰娟/5/nr', '李文亮/5/nr', '应勇/5/nr', '王贺胜/5/nr', '王忠林/5/nr']
    for w in entity_list:
        w_list = w.split('/')
        freq, tag = 1, None
        if len(w_list) > 1:
            freq = w_list[1]
        if len(w_list) > 2:
            tag = w_list[2]
        jieba.add_word(w_list[0], freq=freq, tag=tag)


if __name__ == '__main__':
    addword()
    flag = True
    df = pd.read_excel(r'微博热搜标注.xlsx', dtype=str)
    while flag:
        sub = input("请输入您要分析的主题：")
        df_sub = df[df['主题'].str.contains(sub, na=False)]
        pattern = "2020-01-2[0-9] 12:00:00"
        df_sub = df_sub[df_sub['时间'].str.contains(pattern, na=False)]
        df_sub1 = df_sub.copy()
        df_sub.loc[:, 'spt_word'] = df_sub['热搜内容'].apply(han_word)
        df_co = pd.Series(df_sub['spt_word'].sum()).value_counts()
        dict = df_co.to_dict()
        sorted_values = sorted(dict.values(), reverse=True)[0:20]
        new_dict = {}
        for key, value in dict.items():
            if value in sorted_values:
                new_dict[key] = value
        if new_dict == {0: 1}:
            print("输入的主题不存在，请检查。")
            fl = input("是否继续分析（Y/N）：")
            if fl == 'N' or fl == 'n':
                flag = False
        else:
            maskimg = plt.imread('./mask/facemask.jpg')
            # mask_china = imread('./mask/china.png')
            wc = wordcloud.WordCloud(background_color='white', font_path=r'./smiley/SmileySans-Oblique.ttf', mask=maskimg)
            wc.fit_words(new_dict)
            wc.to_file(r'./cloud/' + sub + '.jpg')
            plt.imshow(wc)
            plt.show()
            print("词云生成成功，请查看。")
            fl = input("是否继续分析（Y/N）：")
            if fl == 'N' or fl == 'n':
                flag = False
