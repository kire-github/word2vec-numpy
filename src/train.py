from typing import Any
import numpy as np

from utils import create_checkpoint
from config import EPOCHS, LEARNING_RATE, NEGATIVE_SAMPLES, BATCH_SIZE, MAKE_CHECKPOINTS, CHECKPOINT_INTERVAL

def train(dataset: Any, model: Any, epochs: int = EPOCHS, lr: float = LEARNING_RATE, num_neg_samples: int = NEGATIVE_SAMPLES) -> None:
    freq = dataset.vocab_freq ** 0.75
    prob = freq / np.sum(freq)

    for epoch in range(epochs):
        pairs = dataset.generate_pairs()
        
        total_loss = 0
        for batch in dataset.batch_generator(pairs, BATCH_SIZE):
            
            # Get centers and contexts
            centers = batch[:, 1]
            contexts = batch[:, 0]

            # Get negative samples
            negative_samples = np.random.choice(
                dataset.vocab_size,
                size=(len(centers),num_neg_samples),
                p=prob
            )

            # Forward pass and update model
            loss_pos, loss_neg, grad_center_pos, grad_center_neg, grad_context_pos, grad_context_neg = model.forward_pass(centers, contexts, negative_samples)
            grad_center = grad_center_pos + grad_center_neg
            model.update(centers, contexts, negative_samples, grad_center, grad_context_pos, grad_context_neg, lr)

            # Normalize embeddings to prevent overflow
            model.normalize_embeddings()

            total_loss += np.sum(loss_pos + loss_neg)

        # Create a checkpoint if enabled
        if MAKE_CHECKPOINTS and (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            create_checkpoint(model, dataset)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    # Final checkpoint after training
    create_checkpoint(model, dataset)