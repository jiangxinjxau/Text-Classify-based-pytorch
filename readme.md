# 基于深度学习Pytorch框架的中文文本分类

## 1、爬虫

首先启动JD.py文件进行爬虫

```python
if __name__ == "__main__":
    list = ['电脑','手机','耳机']
    for item in list:
        spider = JDSpider(item)
        spider.getData(10, 2)
        spider.getData(10, 1)
        spider.getData(10, 0)
```

list列表中是传入的商品类别（如手机、电脑），其中getData的参数是（maxPage, score）

1. maxPage是爬取评论的最大页数，每页10条数据。差评和好评的最大一般页码不相同，一般情况下：好评>>差评>中评    
2. maxPage遇到超出的页码会自动跳出，所以设大点也没有关系。
3. score是指那种评价类型，好评2、中评1、差评0。



运行JD.py，爬取下来的文件存在data/目录下。之后运行ProcessData.py将原始数据集文件进行划分，按8：1：1的比例划分为训练集、数据集和测试集，并将划分后的数据集以csv格式存在dataset/目录下。

![pic_1](https://github.com/NTDXYG/Text-Classify-based-pytorch/blob/master/imgs/pic_1.jpg)

## 2、TorchText使用

TorchText的使用在DataSet.py中

首先是文本预处理

```python
def x_tokenize(x):
    str = re.sub('[^\u4e00-\u9fa5]', "", x)
    return jieba.lcut(str)
```

先通过re正则表达式提取中文，再通过jieba分词将一个句子分成一个列表

```python
TEXT = data.Field(sequential=True, tokenize=x_tokenize,fix_length=100,
            use_vocab=True)
LABEL = data.Field(sequential=False,
            use_vocab=False)
```

对于文本需要做sequential、token、use_vocab处理，对于TEXT还做了一步fix_length的操作，*fix_length*使用此字段的所有示例都将填充到的固定长度,或者对于灵活的序列长度。另外include_length我默认的是False。

之后就是build_vocab和划分iter了。

## 3、模型

- [x] TextCNN
- [x] TextRNN
- [x] TextRNN+Attention
- [x] TextRCNN
- [x] Transformer
- [ ] Some other attention

模型都直接定义在model/目录下，在forward最后返回的out的形状应该是[batch size, num_classes]这样的。

## 4、训练
![pic_1](https://github.com/NTDXYG/Text-Classify-based-pytorch/blob/master/imgs/pic_2.png)

修改完train.py中上述2处直接run就行，训练好的模型将保存在done_model/目录下，默认epoch为10。

## 5、结果比较

### TextCNN

Train loss is 8.4721014514855e-06, Accuracy:0.9388131220051603 

Valid loss:0.0014812756617060203, Accuracy:0.8702064896755162

Test loss:0.0014789274007119663, Accuracy:0.866835229667088

    				precision    recall  f1-score   support
    
              差评       0.82      0.82      0.82       765
              中评       0.81      0.78      0.80       755
              好评       0.95      0.98      0.97       853
    
        accuracy                            0.87      2373
       macro avg        0.86      0.86      0.86      2373
    weighted avg        0.87      0.87      0.87      2373

### TextRNN

Train loss is 1.5317086491512715e-05, Accuracy:0.9159601916697383 

Valid loss:0.0017102944212928383, Accuracy:0.8605141171512853 

Test loss:0.0017361109687766595, Accuracy:0.8512431521281079 

    				precision    recall  f1-score   support
    
              差评       0.78      0.83      0.81       765
              中评       0.77      0.76      0.77       755
              好评       0.99      0.95      0.97       853
    
        accuracy                             0.85      2373
       macro avg        0.85      0.85       0.85      2373
    weighted avg        0.85      0.85       0.85      2373

### TextRCNN

Train loss is 3.0929619438256156e-05, Accuracy:0.943446895898057 

Valid loss:0.0028947687480603396, Accuracy:0.865149599662874 

Test loss:0.0028887669396109083, Accuracy:0.8613569321533924

              precision    recall  f1-score   support
    
          差评       0.80      0.84      0.82       765
          中评       0.80      0.76      0.78       755
          好评       0.97      0.97      0.97       853
    
    accuracy                            0.86       2373
    macro avg       0.86      0.86      0.86       2373
    weighted avg    0.86      0.86      0.86       2373
### TextRNN+Attention

Train loss is 7.891802795062251e-06, Accuracy:0.9661945131904587 

Valid loss:0.0025220917992284414, Accuracy:0.8516645596291614  

Test loss:0.002598140167376184, Accuracy:0.8428150021070375 

    				 precision    recall  f1-score   support
    
              差评       0.80      0.76      0.78       765
              中评       0.76      0.77      0.76       755
              好评       0.95      0.98      0.97       853
    
        accuracy                            0.84      2373
       macro avg        0.84      0.84      0.84      2373
    weighted avg        0.84      0.84      0.84      2373

### Transformer

Train loss is 1.4261148042584926e-05, Accuracy:0.8735716918540362 

Valid loss:0.001806154941336997, Accuracy:0.8529287821323219 

Test loss:0.0016977747457217326, Accuracy:0.8533501896333755 

    				 precision    recall  f1-score   support
    
              差评       0.78      0.86      0.82       765
              中评       0.81      0.72      0.76       755
              好评       0.95      0.96      0.96       853
    
        accuracy                            0.85      2373
       macro avg        0.85      0.85      0.85      2373
    weighted avg        0.85      0.85      0.85      2373

## 6、使用训练好的模型进行文本分类

修改Classify.py中   model = getModel('textrcnn')   ，模型改为之前训练好的模型名称。

然后直接run Classify.py就行

Demo如下

Text:垃圾，这个东西最好别买

Label:0 (差评)
