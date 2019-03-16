# pytorch_text_classification
- A classification task implement in pytorch for my own architecture.

## Requirement ##

	pyorch : 0.3.1
	python : 3.6
	cuda : 8.0/9.0 (support cuda speed up, can chose)

## Usage ##
 
modify the config file, see the Config directory([here](https://github.com/bamtercelboo/pytorch_text_classification/tree/master/Config)) for detail.  

	1、python main.py
	2、python main.py --config_file ./Config/config.cfg --train -p


## Model ##

- CNN
- BiLSTM
- Updating

## Data ##

- SST-Binary

## Result ##

The following test set accuracy are based on the best dev set accuracy.    

| Data/Model | % SST-Binary |  
| ------------ | ------------ |  
| CNN | 84.23 |  
| Bi-LSTM | 86.27 |  


## Reference ##

- [cnn-lstm-bilstm-deepcnn-clstm-in-pytorch](https://github.com/bamtercelboo/cnn-lstm-bilstm-deepcnn-clstm-in-pytorch)
- [基于pytorch的CNN-LSTM神经网络模型调参小结](http://www.cnblogs.com/bamtercelboo/p/7469005.html "基于pytorch的CNN-LSTM神经网络模型调参小结")
- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)
- [Context-Sensitive Lexicon Features for Neural Sentiment Analysis](https://arxiv.org/pdf/1408.5882.pdf)

## Question ##

- if you have any question, you can open a issue or email `bamtercelboo@{gmail.com, 163.com}`.

- if you have any good suggestions, you can PR or email me.