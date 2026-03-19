from typing import Dict, Any, Tuple
import numpy as np
from numpy.typing import NDArray

from config import CHECKPOINT_NAME, EPSILON

def cosine_similarity(vec1: NDArray[np.floating], vec2: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def sigmoid(x: NDArray[np.floating]) -> NDArray[np.floating]:
    x = np.clip(x, -30, 30)
    return 1 / (1 + np.exp(-x))

def positive_step(e_c: NDArray[np.floating], c_p: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    score = np.sum(e_c * c_p, axis=1)
    prob = sigmoid(score)

    loss = -np.log(np.clip(prob, EPSILON, 1 - EPSILON))
    grad_center = (prob - 1)[:, None] * c_p
    grad_context = (prob - 1)[:, None] * e_c

    return loss, grad_center, grad_context

def negative_step(e_c: NDArray[np.floating], c_n: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    score = np.sum(c_n * e_c[:, None, :], axis=2)
    prob = sigmoid(-score)

    loss = -np.sum(np.log(np.clip(prob, EPSILON, 1 - EPSILON)), axis=1)
    grad_center = np.sum(prob[:, :, None] * c_n, axis=1)
    grad_negatives = prob[:, :, None] * e_c[:, None, :]

    return loss, grad_center, grad_negatives

def create_checkpoint(model: Any, dataset: Any) -> None:
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

def load_checkpoint(path: str) -> Tuple[NDArray[np.floating], NDArray[np.floating], Dict[str, int], Dict[int, str]]:
    """
    Load a checkpoint so that it can be restored
    """
    data = np.load(path, allow_pickle=True)

    W_embedding = data["W_embedding"]
    W_context = data["W_context"]
    word_to_idx = data["word_to_idx"].item()
    idx_to_word = data["idx_to_word"].item()

    return W_embedding, W_context, word_to_idx, idx_to_word