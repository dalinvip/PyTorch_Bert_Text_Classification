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



## Bert ##

- use [Bert_Script](https://github.com/bamtercelboo/PyTorch_Bert_Text_Classification/tree/master/Bert_Script) to extract feature from **bert-base-uncased** bert model.

## Model ##

- CNN
- BiLSTM
- BiLSTM + BertFeature
- updating 

## Data ##

- SST-Binary

## Result ##
The following test set accuracy are based on the best dev set accuracy.    

| Model |Bert-Encoder |% SST-Binary |  
| ------------ | ------------ |  ------------ |  
| Bi-LSTM | - |  86.4360 |  
| Bi-LSTM | AvgPooling |  86.3811 |    
| Bi-LSTM | MaxPooling |  86.9303 |  
| Bi-LSTM | BiLSTM+MaxPool |  89.7309 |  

## Reference ##

- [cnn-lstm-bilstm-deepcnn-clstm-in-pytorch](https://github.com/bamtercelboo/cnn-lstm-bilstm-deepcnn-clstm-in-pytorch)

- https://github.com/huggingface/pytorch-pretrained-BERT  

- https://github.com/google-research/bert  

## Question ##

- if you have any question, you can open a issue or email **bamtercelboo@{gmail.com, 163.com}**.

- if you have any good suggestions, you can PR or email me.