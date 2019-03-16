# PyTorch_Bert_Text_Classification
- Bert For Text Classification in SST  

## Requirement ##

	PyTorch : 1.0.1
	Python : 3.6
	Cuda : 9.0 (support cuda speed up, can chose)

## Usage ##
 
modify the config file, see the Config directory.

	1、sh run_train_p.sh
	2、python -u main.py --config ./Config/config.cfg --device cuda:0 --train -p


## Model ##

- CNN
- BiLSTM
- BiLSTM + BertFeature
- updating 

## Data ##

- SST-Binary

## Result ##


## Reference ##

- [cnn-lstm-bilstm-deepcnn-clstm-in-pytorch](https://github.com/bamtercelboo/cnn-lstm-bilstm-deepcnn-clstm-in-pytorch)
- [基于pytorch的CNN-LSTM神经网络模型调参小结](http://www.cnblogs.com/bamtercelboo/p/7469005.html "基于pytorch的CNN-LSTM神经网络模型调参小结")
- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)
- [Context-Sensitive Lexicon Features for Neural Sentiment Analysis](https://arxiv.org/pdf/1408.5882.pdf)

## Question ##

- if you have any question, you can open a issue or email **bamtercelboo@{gmail.com, 163.com}**.

- if you have any good suggestions, you can PR or email me.