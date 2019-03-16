## Role of document in `DataUtils` directory ##
- DataUtils
	-  `Alphabet.py`  ------ Build vocab by train data or dev/test data

	- `Batch_Iterator.py` ------ Build batch and iterator for train/dev/test data, get train/dev/test iterator

	- `Common.py` ------ The file contains some common attribute, like random seeds, padding, unk and others

	- `Embed.py`  ------ Loading Pre-trained word embedding( `glove` or `word2vec` ), `zeros，avg, uniform, nn.Embedding for OOV`.

	-  `Optim.py` ------ Encapsulate the `optimizer`.

	-  `Pickle.py` ------ Encapsulate the `pickle`.

	-  `utils.py` ------ common function.

	-  `mainHelp.py` ------ main help file.
