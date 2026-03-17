import numpy as np
import config
from utils import create_checkpoint

def train(dataset, model, epochs=config.EPOCHS, lr=config.LEARNING_RATE, num_neg_samples=config.NEGATIVE_SAMPLES):
    freq = dataset.vocab_freq ** 0.75
    prob = freq / np.sum(freq)

    for epoch in range(epochs):
        pairs = dataset.generate_pairs()
        
        total_loss = 0
        for center, context in pairs:
            # Get negative samples
            negative_samples = np.random.choice(dataset.vocab_size, num_neg_samples, p=prob)

            # Forward pass and update model
            loss_pos, loss_neg, grad_center_pos, grad_center_neg, grad_context_pos, grad_context_neg = model.forward_pass(center, context, negative_samples)

            model.update(center, context, negative_samples, grad_center_pos, grad_center_neg, grad_context_pos, grad_context_neg, lr)
            total_loss += loss_pos + loss_neg
        
        # Create a checkpoint if enabled
        if config.MAKE_CHECKPOINTS and (epoch + 1) % config.CHECKPOINT_INTERVAL == 0:
            create_checkpoint(model, dataset)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    # Final checkpoint after training
    create_checkpoint(model, dataset)