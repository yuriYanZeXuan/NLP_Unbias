# NLP文本表征课程大作业仓库

## 实现思路
1. 输入数据 -> 调API改词
2. 产生新文本和老文本，进行情感分析
3. 将新文本和老文本输入训练模型，测量语义准确率
4. 比较情感系数和准确率
   1. 情感更加接近neutral
   2. 准确率损失在一定水平内

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
