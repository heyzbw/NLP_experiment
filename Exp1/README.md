## 实验一：分词

- `bi_freq.py`构建获取二元组频率词典
  > 添加二元组频率字典提升打榜分（F1-值）约 0.2
- `get_token.py`获取key为人名的item
  > 词典添加`jieba`分出的人名提升打榜分（F1-值）约 0.5
  > 
  > 词典添加`hanlp`分出的人名提升打榜分（F1-值）约 1.2
- `getvo_freq.py`从train.csv获取带频数的词典
- `main.py`最初实现的最大匹配算法（未使用频数）
  > 运行前先运行`get_token.py`和`getvo_freq.py`获取词典
  > 
  > 打榜得分（F1-值） 92.917
- `main_freq`使用互信息与二元组频率字典改进的最大匹配算法
  > 运行前先运行`bi_freq.py`、`get_token.py`和`getvo_freq.py`获取词典
  >
  > 打榜得分（F1-值） 94.186
- `main_tool`使用分词库完成
  > 使用`jieba`，打榜分（F1-值）为 80.401
  >
  > 使用`hanlp`，打榜分（F1-值）为 97.979

注*：所有不符合规定的实验性质的提交都由本人小号（`TEAM NAME`为`GOLD`）完成

#### [打榜最终成绩（F1-值）](https://www.kaggle.com/competitions/csu-ai-inclass-nlp-2023/leaderboard?tab=public) `Public: 94.186` `Private: 94.309`
