import numpy as np
from config import CHECKPOINT_NAME

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def positive_step(e_c, c_p):
    score = np.dot(c_p, e_c)
    prob = sigmoid(score)

    loss = -np.log(prob)
    grad_center = (prob - 1) * c_p
    grad_context = (prob - 1) * e_c

    return loss, grad_center, grad_context

def negative_step(e_c, c_n):
    score = c_n @ e_c
    prob = sigmoid(score)

    loss = -np.sum(np.log(1 - prob))
    grad_center = np.sum(prob[:, None] * c_n, axis=0)
    grad_negatives = prob[:, None] * e_c

    return loss, grad_center, grad_negatives

def create_checkpoint(model, dataset):
    """
    Creates a checkpoint from which training can continue
    """
    np.savez(
        CHECKPOINT_NAME,
        W_embedding=model.W_embedding,
        W_context=model.W_context,
        word_to_idx=dataset.word_to_idx,
        idx_to_word=dataset.idx_to_word
    )