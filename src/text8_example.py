from utils import cosine_similarity
from dataset import Dataset
from model import SGNS
from train import train

def most_similar(word, dataset, model,number=5):
    if word not in dataset.word_to_idx:
        return []
    idx = dataset.word_to_idx[word]
    vec = model.W_embedding[idx]
    sims = cosine_similarity(model.W_embedding, vec)
    nearest = sims.argsort()[-number-1:-1][::-1]
    return [dataset.idx_to_word[i] for i in nearest]


if __name__ == "__main__":
    # ----------------------
    # 1. Load and preprocess vocab
    # ----------------------
    print("Processing dataset...")

    with open("text8", "r") as f:
        text = f.read()

    dataset = Dataset(text)

    print("Dataset initialized")

    # ----------------------
    # 2. Initialize model
    # ----------------------
    model = SGNS(vocab_size=dataset.vocab_size)

    # ----------------------
    # 3. Training loop
    # ----------------------
    print("Starting training...")
    train(dataset, model)
    print("Training completed")

    # ----------------------
    # 4. Test
    # ----------------------
    print("Testing embeddings...")
    print("Most similar to 'anarchism':", most_similar("anarchism", dataset, model))
    print("Most similar to 'term':", most_similar("term", dataset, model))
    print("Most similar to 'class':", most_similar("class", dataset, model))