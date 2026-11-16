import torch

def softmax(t, dim):
    """Numerically stable softmax along dim.

    Args:
        t (torch.Tensor): input tensor
        dim (int): dimension along which to apply softmax

    Returns:
        torch.Tensor: tensor of same shape as t; slices along dim sum to 1
    """
    # TODO: subtract max along dim, exp, then normalize
    max_z = torch.max(t, dim, keepdim=True).values
    norm_z = t - max_z
    softmax_values = torch.exp(norm_z) / torch.sum(torch.exp(norm_z), dim, keepdim=True)
    return softmax_values
