import numpy as np
from utils import positive_step, negative_step
from config import EMBEDDING_DIM

class SGNS:
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Matrices for embedding (center) and context
        self.W_embedding = np.random.rand(vocab_size, embedding_dim) * 0.01
        self.W_context = np.random.rand(vocab_size, embedding_dim) * 0.01
    
    def forward_pass(self, center_idx, positive_idx, negative_indices):
        """
        Compute the forward pass for a single training example
        """  
        e_c = self.W_embedding[center_idx]
        c_p = self.W_context[positive_idx]
        c_n = self.W_context[negative_indices]

        loss_pos, grad_center_pos, grad_context_pos = positive_step(e_c, c_p)
        loss_neg, grad_center_neg, grad_context_neg = negative_step(e_c, c_n)

        return loss_pos, loss_neg, grad_center_pos, grad_center_neg, grad_context_pos, grad_context_neg
    
    def update(self, center_idx, positive_idx, negative_indices, grad_center, grad_context_pos, grad_context_neg, learning_rate):
        """
        Update the weights of the model
        """
        np.add.at(self.W_embedding, center_idx, -learning_rate * grad_center)
        np.add.at(self.W_context, positive_idx, -learning_rate * grad_context_pos)

        _, num_negatives = negative_indices.shape
        for i in range(num_negatives):
            np.add.at(self.W_context, negative_indices[:, i], -learning_rate * grad_context_neg[:, i])

    def normalize_embeddings(self):
        """
        Normalize the embedding matrix to prevent overflow
        """
        norms = np.linalg.norm(self.W_embedding, axis=1, keepdims=True)
        self.W_embedding /= np.maximum(norms, 1.0)

        norms = np.linalg.norm(self.W_context, axis=1, keepdims=True)
        self.W_context /= np.maximum(norms, 1.0)