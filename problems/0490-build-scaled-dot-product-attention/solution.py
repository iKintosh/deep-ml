import numpy as np

def softmax(z: np.ndarray) -> np.ndarray:
    max_z = np.max(z, 1)
    norm_z = np.exp(np.transpose(z) - max_z)
    return np.transpose(norm_z / np.sum(norm_z, 0))

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
    """
    Compute Scaled Dot-Product Attention.
    
    Args:
        Q: Query matrix of shape (seq_len_q, d_k)
        K: Key matrix of shape (seq_len_k, d_k)
        V: Value matrix of shape (seq_len_k, d_v)
        mask: Optional binary mask of shape (seq_len_q, seq_len_k)
    
    Returns:
        Tuple of (output, attention_weights)
    """
    # Your code here
    K_T = np.transpose(K)
    attention = np.matmul(Q, K_T) / np.sqrt(K.shape[1])
    if mask is not None:
        attention[mask==0] = -np.inf
    attention = softmax(attention)
    output = np.matmul(attention, V)
    return output, attention