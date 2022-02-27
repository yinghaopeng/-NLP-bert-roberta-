使用paddlepaddle框架完成对中文类型垃圾邮件进行分类
#  1、项目背景介绍
在我们日常生活中，我们使用邮件进行信息传递，就好比使用QQ、微信等聊天软件一样进行双方隔空对话，但是邮箱其实对于我们来说其实更大的作用，在于我们和对方进行一些重要信息沟通，但是往往在我们的邮箱时不时存在一些垃圾邮件，而这些垃圾邮件的来源可以说是，五花八门，因为以笔者自身的例子作为一个案例分析，笔者在中考后购买了苹果的手机，当时就是使用自己的QQ邮箱进行注册，自从那天开始，后面苹果每逢周一或者是遇到新的产品上市，或者是新的游戏上市app store，则会以邮件的方式推送给我，对于用户的我来说，我觉得无疑是会影响我的邮箱使用，再者当收到邮件时候，邮箱给我发送邮件提醒，甚至会误导我以为是一些重要的邮件，又或者笔者之前才加coursera的课程，当时也是用QQ邮箱进行注册登录，后面也是不定时的给我发送一些推销它们产品的广告，我觉得这种垃圾邮件一来会对邮箱的容量进行占据，另外一方面会让使用者降低对邮件的使用频率，因此笔者以垃圾邮件的数据集作为本次项目的训练，并希望日后能够用上部署在一些邮箱软件上，可以让软件自动帮我们识别发过来的邮件信息，并进行一定的过滤

# 2、数据介绍
[TREC2005-2007垃圾邮件数据集](https://aistudio.baidu.com/aistudio/datasetdetail/89631/1)，原数据集描述：是一个公开的垃圾邮件语料库，由国际文本检索会议提供，分为英文数据集（trec06p）和中文数据集（trec06c），其中所含的邮件均来源于真实邮件保留了邮件的原有格式和内容。除TREC 2006外，还有TREC 2005和TREC 2007的英文垃圾邮件数据集，因为本文主要应对的还是对于中文邮件，因此主要是使用垃圾邮件的中文数据集trec06c作为研究对象，也可以从官网上获取其[数据](https://plg.uwaterloo.ca/~gvcormac/treccorpus06/about.html)

# 3、模型介绍
本文一共使用两个模型进行对比训练，一个是bert模型一个是roberta模型进行对比训练，通过visualdl可视化工具对两种的模型进行观察，并给出哪一种模型较优
## 3.1bert模型
### 3.1.1什么是bert？
BERT的全称为Bidirectional Encoder Representation from Transformers，是一个预训练的语言表征模型。

它强调了不再像以往一样采用传统的单向语言模型或者把两个单向语言模型进行浅层拼接的方法进行预训练，而是采用新的masked language model（MLM），因此能生成深度的双向语言表征。

该模型有以下主要优点：

1）采用MLM对双向的Transformers进行预训练，以生成深层的双向语言表征。

2）预训练后，只需要添加一个额外的输出层进行fine-tune，就可以在各种各样的下游任务中取得state-of-the-art的表现。在这过程中并不需要对BERT进行任务特定的结构修改。
### 3.1.2bert模型结构
以往的预训练模型的结构会受到单向语言模型（从左到右或者从右到左）的限制，因而也限制了模型的表征能力，使其只能获取单方向的上下文信息。

而BERT利用MLM进行预训练并且采用深层的双向Transformer组件来构建整个模型，因此最终生成能融合左右上下文信息的深层双向语言表征。
> 注：单向的Transformer一般被称为Transformer decoder，其每一个token（符号）只会attend到目前往左的token。而双向的Transformer则被称为Transformer encoder，其每一个token会attend到所有的token。

<div align=center><img src="https://ai-studio-static-online.cdn.bcebos.com/31eb385fe74d4466a2ca9ae0a279ac5a3c4ac29b823142c99b220d0c3a0579dd" width="  "></div>
<center>Transformers模型结构</center>

Transformer进行堆叠，形成一个更深的神经网络，如下图所示
<div align=center><img src="https://ai-studio-static-online.cdn.bcebos.com/e27076146565415395af046993f39c87177f0552fa154b248c1ad93569c5cfc5" width="  "></div>
<center>对Transformers进行堆叠</center>

最终，经过多层Transformer的堆叠后bert的主体如下所示
<div align=center><img src="https://ai-studio-static-online.cdn.bcebos.com/d33407faede14fc49908c515c72b763e588e1ca569b343e8a85949ea71379c00" width="  "></div>
<center>bert主体结构</center>

使用visualdl可视化工具对模型训练的时候进行可视化操作，通过观察其所需时间，算法的收敛速度，以及其准确率的大小，方便和后面所使用的roberta模型进行对比
<div align=center><img src="https://ai-studio-static-online.cdn.bcebos.com/db856685ef134cd59e15e402b5d084ecc2fb0e43ece94bdca263c50808d84cbc" width="  "></div>
<center>使用bert模型训练</center>

## 3.2roberta模型
### 3.2.1 roberta模型是什么
[roberta模型论文](https://arxiv.org/abs/1907.11692)可以在这里下载到roberta算法的论文，同时[roberta算法](https://github.com/brightmart/roberta_zh)在github上已经有了开源的仓库

roberta是bert的改进版，通过改进训练任务和数据生成方式、训练更久、使用更大批次、使用更多数据等获得了SOTA的效果

roberta算法的改进如下
- More data(更多的数据)
> 文章基于 BERT 提出了一种效果更好的预训练模型训练方式，其主要的区别如下： 训练数据上，RoBERTa 采用了 160G 的训练文本，而 BERT 仅使用 16G 的训练文本。

<div align=center><img src="https://ai-studio-static-online.cdn.bcebos.com/eb138f1e814e4707b5d1cda1fa5c5b8a17e6d162cdb949f1bed00c901dd720c1" width="  "></div>
<center>不同算法预训练数据量对比</center>

- More Steps(更多训练)
<div align=center><img src="https://ai-studio-static-online.cdn.bcebos.com/876bd97d28d54aefae52f234a3a074537f3e7373c7e04c24a3a9ec5828183976" width="  "></div>

- Large Batch(更大批次)
> 批量（batch），常规设置128，256等等便可，如 BERT则是256，RoBERTa 在训练过程中使用了更大的批数量。研究人员尝试过从 256 到 8000 不等的批数量。

- Adam optimizer

Adam借鉴了Kingma等人的改进，使用$\beta_1=0.9$、$\beta_2=0.999$、$\epsilon=1e-6$,并且$L_2$的衰减权重设置为$0.01$，在前10000$steps$是warmed up学习率是$1e-4$,并且是线性的衰减，所有层和Attention权重的dropout=0.1，预训练模型训练1,000,000steps最小batch256，最大batch512

<div align=center><img src="https://ai-studio-static-online.cdn.bcebos.com/da42f0ccea414bfcbb8be6d7362514586623311774d74417a78553816a4fd004" width="  "></div>

<center>Transformer使用的warmed up学习率</center>

- Next Sentence Prediction
> Next Sentence Prediction (NSP) 数据生成方式和任务改进：取消下一个句子预测，并且数据连续从一个文档中获得
### 3.2.2roberta模型结构
<div align=center><img src="https://ai-studio-static-online.cdn.bcebos.com/cf165061c4164fcaa6599e04501e44e2d9e43cc9c1834340ba8b8d9f5ffb061f" width="  "></div>

通过使用可视化工具可视化，可以看出效果如下图所示
<div align=center><img src="https://ai-studio-static-online.cdn.bcebos.com/30184e47812e4f088f33908d5877e036d2533d5bdc2945c8b96a73e8f6965793" width="  "></div>
<center>使用roberta模型训练</center>

# 4、总结与升华
通过自己动手去实现这个项目后，针对于之前只能通过降低库函数版本以及框架来实现的算法，现在可以做到根据现有最新的框架去实现自己想要的算法，同时本文完成的是关于中文垃圾邮件的分类问题，使用的是bert模型和roberta模型，网上有人评论则提到，这两个模型的不同之处在于后者其实是使用名副其实的暴力调参法，在跑实验的过程中，也可以看出，使用bert模型去训练数据的话，其训练速度比较慢，如下图所示在其性能监控中可以看出
<div align=center><img src="https://ai-studio-static-online.cdn.bcebos.com/ba6d1473cf004d8e8bdf204d8ac29e0fa6f9dfc59a8d44c6b9a872a8ed90cc41" width="  "></div>
<center>使用bert算法性能监控</center>

而且所需花费的时间也很长，相反，若使用的是roberta，其算法的收敛速度，训练速度相比于bert来说都有一定的改进，如下图所示
<div align=center><img src="https://ai-studio-static-online.cdn.bcebos.com/dc3dc4c52d3042158b3f328814445fdcaf3b276bf3d540bbb94067eeaaaf0671" width="  "></div>
<center>使用roberta算法性能监控</center>

从而我们可以得出一个结论，就是如果我们有充裕的时间的话，可以使用bert模型进行训练数据，倘若我们想比较快的能够显示出结果，那么我们可以使用roberta来进行算法的实现，因为其两者的准确率在epoch达到10次以上后，其实两者的准确率都相当的高
