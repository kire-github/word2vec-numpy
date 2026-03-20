# Profiling

Performance profiling was done using cProfile, the current performance can be analyzed by running [`profiler.py`](./src/profiler.py).

## Pre-optimizations

```
8642099 function calls (38641752 primitive calls) in 124.738 seconds

Ordered by: cumulative time
List reduced from 462 to 30 due to restriction <30>

ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    1    0.000    0.000  124.738  124.738 /home/admin/code/word2vec-numpy/src/text8_example.py:16(text8_example)
    1    6.530    6.530  121.340  121.340 /home/admin/code/word2vec-numpy/src/train.py:7(train)
21554    5.352    0.000   52.085    0.002 /home/admin/code/word2vec-numpy/src/model.py:30(update)
150878   46.734    0.000   46.734    0.000 {method 'at' of 'numpy.ufunc' objects}
21554    4.107    0.000   31.028    0.001 /home/admin/code/word2vec-numpy/src/model.py:17(forward_pass)
21554   14.606    0.001   22.247    0.001 /home/admin/code/word2vec-numpy/src/utils.py:24(negative_step)
21554    8.883    0.000   21.766    0.001 /home/admin/code/word2vec-numpy/src/model.py:41(normalize_embeddings)
```

The most important insights gathered from this are that `numpy.at` is being called much more times than what is minimal for `update()` and `forward_pass()`, and that `normalize_embeddings()` takes around 21 seconds of the total runtime, however if we don't normalize the embeddings after every batch, there is a chance they will explode. This is a performance trade-off worth taking.

## Fixes

### numpy.at
To address the issue of the `numpy.at` calls, adding the negative steps to the embeddings was changed from:

```py
_, num_negatives = negative_indices.shape
for i in range(num_negatives):
    np.add.at(self.W_context, negative_indices[:, i], -learning_rate * grad_context_neg[:, i])
```

to:
```py
flat_indices = negative_indices.flatten()
flat_grads = grad_context_neg.reshape(-1, self.embedding_dim)
np.add.at(self.W_context, flat_indices, -learning_rate * flat_grads)
```

This removed the individual calls of `numpy.at` for each negative example for the batches, and instead flattens the indices and gradients, so that they can be applied in a single call.

## Post-optimizations

```
38598991 function calls (38598644 primitive calls) in 128.393 seconds

Ordered by: cumulative time
List reduced from 464 to 30 due to restriction <30>

ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    1    0.000    0.000  128.393  128.393 /home/admin/code/word2vec-numpy/src/text8_example.py:16(text8_example)
    1    6.839    6.839  124.878  124.878 /home/admin/code/word2vec-numpy/src/train.py:7(train)
21554    4.017    0.000   51.909    0.002 /home/admin/code/word2vec-numpy/src/model.py:30(update)
64662   47.698    0.001   47.698    0.001 {method 'at' of 'numpy.ufunc' objects}
21554    4.493    0.000   33.169    0.002 /home/admin/code/word2vec-numpy/src/model.py:17(forward_pass)
21554   15.481    0.001   23.679    0.001 /home/admin/code/word2vec-numpy/src/utils.py:24(negative_step)
21554    9.275    0.000   22.482    0.001 /home/admin/code/word2vec-numpy/src/model.py:41(normalize_embeddings)
```

We see that the number of `numpy.at` calls reduced to the correct amount, however they ultimately take longer than the original version, thus the changes were reverted.

# Profiling parameters

text8 dataset restricted to the first 100k tokens.

[`config.py`](./src/config.py)
```py
# dataset parameters
WINDOW_SIZE = 5                     # context window size
MIN_WORD_FREQUENCY = 5              # minimum frequency for a word to be included in the vocab
NEGATIVE_SAMPLES = 5                # number of negative samples for each positive pair

# training parameters
EPOCHS = 13                         # number of training epochs
LEARNING_RATE = 0.001               # learning rate
EMBEDDING_DIM = 100                 # dimensionality of the word embeddings
BATCH_SIZE = 512                    # batch size for batch processing
EPSILON = 1e-10                     # small value to prevent division by zero

# checkpointing
MAKE_CHECKPOINTS = True             # whether to create checkpoints during training
CHECKPOINT_INTERVAL = 2             # how many epochs between checkpoints, should be > 0
CHECKPOINT_NAME = "sgns_save.npz"   # filename for the checkpoint, should end with .npz
``