# word2vec-numpy

This project implements the Word2Vec algorithm using a Skipgram model with negative sampling in NumPy without the use of an ML framework.

## Usage

The training's parameters can be adjusted in [`config.py`](./config.py).

An example of how to train the model on the [`text8`](https://mattmahoney.net/dc/textdata.html) dataset is available in [`text8_example.py`](./text8_example.py). To be able to run this, you must download and unzip the [`text8`](https://mattmahoney.net/dc/textdata.html) dataset. This can be done using:

```
wget http://mattmahoney.net/dc/text8.zip
unzip text8.zip
```

Following this, it can be run via:

```
python ./text8_example.py
```

## References & Citations

- [**The Illustrated Word2Vec**](https://jalammar.github.io/illustrated-word2vec/)
- [**word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method**](https://arxiv.org/abs/1402.3722)
- [**word2vec Parameter Learning Explained**](https://arxiv.org/abs/1411.2738)

