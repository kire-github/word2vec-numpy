import numpy as np
from utils import positive_step, negative_step

class SGNS:
    def __init__(self, vocab_size, embedding_dim):
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
    
    def update(self, center_idx, positive_idx, negative_indices, grad_center_pos, grad_center_neg, grad_context_pos, grad_context_neg, learning_rate):
        """
        Update the weights of the model
        """
        self.W_embedding[center_idx] -= learning_rate * (grad_center_pos + grad_center_neg)
        self.W_context[positive_idx] -= learning_rate * grad_context_pos
        self.W_context[negative_indices] -= learning_rate * grad_context_neg