# NLP文本表征课程大作业仓库

## 实现思路
1. 输入数据 -> 调API改词
2. 产生新文本和老文本，进行情感分析
3. 将新文本和老文本输入训练模型，测量语义准确率
4. 比较情感系数和准确率
   1. 情感更加接近neutral
   2. 准确率损失在一定水平内
   

## 情感极性判断

### nltk使用:

compound计算方法参见:https://www.coder.work/article/96459
省流:

compound 分数是 sum_s 和的归一化分数

sum_s 是根据一些启发式算法和情感词典(又名情感强度)计算的加权平均数

归一化分数只是 sum_s 除以其平方加上一个增加归一化函数分母的 alpha 参数。

### Baidu api使用:

``` shell
pip install baidu-aip
pip install chardet
```


``` python
from aip import AipNlp
APP_ID="34711754"
API_KEY="O3fr7AjA7XSkukChkdRSic7y"
SECRET_KEY="Btyq01fqQ7vf6EYU0xpaOunARoBU38B0"

client=AipNlp(APP_ID,API_KEY,SECRET_KEY)
client.sentimentClassify(text)
```

文本内容（GBK编码），最大2048字节

情感倾向分析 返回数据参数详情

|参数 	|是否必须 |	类型 |	说明|
|---|----|---|---|
|text |	是 	|string |	输入的文本内容|
|items |	是 	|array 	|输入的词列表|
|+sentiment |	是 |	number |	表示情感极性分类结果, 0:负向，1:中性，2:正向|
|+confidence |	是 |	number |	表示分类的置信度|
|+positive_prob |	是 |	number |	表示属于积极类别的概率|
|+negative_prob |	是 |	number |	表示属于消极类别的概率|

## 数据集

电影评论数据集

=======

Data Format Summary （from rt-polaritydata.README.1.0）

- rt-polaritydata.tar.gz: contains this readme and two data files that
  were used in the experiments described in Pang/Lee ACL 2005.

  Specifically: 
  * rt-polarity.pos contains 5331 positive snippets
  * rt-polarity.neg contains 5331 negative snippets

  Each line in these two files corresponds to a single snippet (usually
  containing roughly one single sentence); all snippets are down-cased.  
  The snippets were labeled automatically, as described below (see 
  section "Label Decision").

  Note: The original source files from which the data in
  rt-polaritydata.tar.gz was derived can be found in the subjective
  part (Rotten Tomatoes pages) of subjectivity_html.tar.gz (released 
  with subjectivity dataset v1.0).

=======
