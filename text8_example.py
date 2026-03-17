from utils import cosine_similarity
from dataset import Dataset
from model import SGNS
from train import train

# ----------------------
# 1. Load and preprocess vocab
# ----------------------
print("Processing dataset...")

with open("text8", "r") as f:
    text = f.read()

window_size = 5
min_frequency = 5
dataset = Dataset(text, window_size, min_frequency)

print("Dataset initialized")

# ----------------------
# 2. Initialize model
# ----------------------
embedding_dim = 50
model = SGNS(vocab_size=dataset.vocab_size, embedding_dim=embedding_dim)

# ----------------------
# 3. Training loop
# ----------------------
epochs = 20
lr = 0.02
neg_samples = 5

print("Starting training...")
train(dataset, model, epochs=epochs, lr=lr, num_neg_samples=neg_samples)
print("Training completed")

# ----------------------
# 4. Test
# ----------------------

print("Testing embeddings...")
def most_similar(word, number=5):
    if word not in dataset.word_to_idx:
        return []
    idx = dataset.word_to_idx[word]
    vec = model.W_embedding[idx]
    sims = cosine_similarity(model.W_embedding, vec)
    nearest = sims.argsort()[-number-1:-1][::-1]
    return [dataset.idx_to_word[i] for i in nearest]

# Example queries
print("Most similar to 'anarchism':", most_similar("anarchism"))
print("Most similar to 'term':", most_similar("term"))
print("Most similar to 'class':", most_similar("class"))