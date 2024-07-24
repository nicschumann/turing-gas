from torch import Tensor
from jaxtyping import Int

from math import floor

import torch

torch.manual_seed(1)


def select(
    soup: Int[Tensor, "programs length data"],
    *,
    selected_percentage: float = 1.0,
    random_state: torch.Generator,
) -> tuple[Int[Tensor, "programs length data"], Int[Tensor, "indices"]]:
    """Given a current state for the soup, randomly select pairs from the
    soup, concatenate them, and return them, as well as their original indices,
    so they can be returned from the soup in the correct positions.
    """
    num_programs, program_length, _ = soup.shape

    assert (
        num_programs % 2 == 0
    ), f"Must contain an even number of programs for selection, but received {num_programs}"

    cutoff = floor(num_programs * selected_percentage)
    cutoff += cutoff % 2  # make sure it's an even number

    # get random, non-repeating indices, up to the proportion
    # of the soup that we want to sample.
    random_indices = torch.randperm(num_programs, generator=random_state)[:cutoff]

    # index the soup with the random indices, and reshape so that
    # the programs are concatenated
    paired_soup = soup[random_indices].reshape(
        num_programs // 2, 2 * program_length, -1
    )

    return paired_soup, random_indices
