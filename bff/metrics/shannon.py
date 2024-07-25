import torch
import brotli

from ..interp import CHAR_IDX


def entropy(soup: torch.Tensor, *, instruction_space_size: int):
    flattened_soup = soup[:, :, CHAR_IDX].flatten()
    freqs = (
        torch.bincount(flattened_soup, minlength=instruction_space_size)
        / flattened_soup.shape[0]
    )

    # NOTE(Nic): sanity: for a uniform distribution over symbols,
    # this should be ~= -ln(1/256) = ln(256)
    shannon_entropy = torch.distributions.Categorical(probs=freqs).entropy()

    uncompressed = flattened_soup.numpy().tobytes()
    compressed = brotli.compress(flattened_soup.numpy().tobytes())

    kolmogorov_complexity = len(uncompressed) / len(compressed)

    print(f"shannon: {shannon_entropy}")
    print(f"kolmogorov: {kolmogorov_complexity}")
    print(f"higher order entropy: {shannon_entropy - kolmogorov_complexity}")

    # torch.distributions.Categorical(probs=)
