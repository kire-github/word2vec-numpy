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