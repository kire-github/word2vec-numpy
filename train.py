import numpy as np

def train(dataset, model, epochs=1, lr=0.01, num_neg_samples=5):
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

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")