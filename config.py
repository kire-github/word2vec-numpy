# dataset parameters
WINDOW_SIZE = 5
MIN_WORD_FREQUENCY = 5
NEGATIVE_SAMPLES = 5

# training parameters
EPOCHS = 5
LEARNING_RATE = 0.02
EMBEDDING_DIM = 50

# checkpointing
MAKE_CHECKPOINTS = True
CHECKPOINT_INTERVAL = 2 # should be > 0
CHECKPOINT_NAME = "sgns_save.npz"