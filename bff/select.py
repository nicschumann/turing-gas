from torch import Tensor
from jaxtyping import Int

from math import floor

import torch

torch.manual_seed(1)


def select_pairs(
    soup: Int[Tensor, "programs length data"],
    *,
    selected_percentage: float = 1.0,
    random_state: torch.Generator,
    device: str,
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
    random_indices = torch.randperm(
        num_programs, generator=random_state, device=device
    )[:cutoff]

    # index the soup with the random indices, and reshape so that
    # the programs are concatenated
    paired_soup = soup[random_indices].reshape(
        num_programs // 2, 2 * program_length, -1
    )

    return paired_soup, random_indices


def return_pairs(
    paired_programs: Int[Tensor, "programs doubled_length data"],
    random_indices: Int[Tensor, "indices"],
    soup: Int[Tensor, "programs length data"],
    *,
    device: str,
):
    num_programs, program_length, _ = soup.shape
    num_paired_programs, paired_program_length, _ = paired_programs.shape

    assert (
        num_programs == 2 * num_paired_programs
        and program_length == paired_program_length // 2
    ), f"p_len ({program_length}) must equal paired p_len // 2 ({paired_program_length})"

    soup[random_indices] = paired_programs.reshape(num_programs, program_length, 3)
