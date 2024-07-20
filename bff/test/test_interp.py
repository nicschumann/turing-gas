import pytest
import torch
from dataclasses import dataclass

from bff.interp import step


@dataclass
class State:
    soup: list[list[list[int]]]  # NOTE(Nic): list[list[tuple[int, int, int]]]
    data: list[list[int]]  # NOTE(Nic): list[tuple[int, int, int]]
    running: list[bool]


tests = [
    (
        State(
            soup=[[[0, 0, 1], [0, 0, 0], [0, 0, 0]]], data=[[0, 0, 0]], running=[True]
        ),
        State(
            soup=[[[0, 0, 1], [0, 0, 0], [0, 0, 0]]], data=[[1, 2, 0]], running=[True]
        ),
    ),
    (
        State(
            soup=[[[0, 0, 2], [0, 0, 0], [0, 0, 0]]], data=[[0, 0, 0]], running=[True]
        ),
        State(
            soup=[[[0, 0, 2], [0, 0, 0], [0, 0, 0]]], data=[[1, 1, 0]], running=[True]
        ),
    ),
    (
        State(
            soup=[[[0, 0, 3], [0, 0, 0], [0, 0, 0]]], data=[[0, 0, 0]], running=[True]
        ),
        State(
            soup=[[[0, 0, 3], [0, 0, 0], [0, 0, 0]]], data=[[1, 0, 2]], running=[True]
        ),
    ),
    (
        State(
            soup=[[[0, 0, 4], [0, 0, 0], [0, 0, 0]]], data=[[0, 0, 0]], running=[True]
        ),
        State(
            soup=[[[0, 0, 4], [0, 0, 0], [0, 0, 0]]], data=[[1, 0, 1]], running=[True]
        ),
    ),
    (
        State(
            soup=[[[0, 0, 5], [0, 0, 0], [0, 0, 0]]], data=[[0, 1, 0]], running=[True]
        ),
        State(
            soup=[[[0, 0, 5], [0, 0, 255], [0, 0, 0]]], data=[[1, 1, 0]], running=[True]
        ),
    ),
    (
        State(
            soup=[[[0, 0, 6], [0, 0, 0], [0, 0, 0]]], data=[[0, 1, 0]], running=[True]
        ),
        State(
            soup=[[[0, 0, 6], [0, 0, 1], [0, 0, 0]]], data=[[1, 1, 0]], running=[True]
        ),
    ),
    (
        State(
            soup=[[[0, 0, 7], [0, 0, 4], [0, 0, 5]]], data=[[0, 1, 2]], running=[True]
        ),
        State(
            soup=[[[0, 0, 7], [0, 0, 4], [0, 0, 4]]], data=[[1, 1, 2]], running=[True]
        ),
    ),
    (
        State(
            soup=[[[0, 0, 8], [0, 0, 4], [0, 0, 5]]], data=[[0, 1, 2]], running=[True]
        ),
        State(
            soup=[[[0, 0, 8], [0, 0, 5], [0, 0, 5]]], data=[[1, 1, 2]], running=[True]
        ),
    ),
    (
        State(
            soup=[[[0, 0, 9], [0, 0, 0], [0, 0, 10]]], data=[[0, 1, 0]], running=[True]
        ),
        State(
            soup=[[[0, 0, 9], [0, 0, 0], [0, 0, 10]]], data=[[2, 1, 0]], running=[True]
        ),
    ),
    (
        State(
            soup=[[[0, 0, 9], [0, 0, 0], [0, 0, 0]]], data=[[0, 1, 0]], running=[True]
        ),
        State(
            soup=[[[0, 0, 9], [0, 0, 0], [0, 0, 0]]], data=[[0, 1, 0]], running=[False]
        ),
    ),
    (
        State(
            soup=[[[0, 0, 9], [0, 0, 1], [0, 0, 0]]], data=[[0, 1, 0]], running=[True]
        ),
        State(
            soup=[[[0, 0, 9], [0, 0, 1], [0, 0, 0]]], data=[[1, 1, 0]], running=[True]
        ),
    ),
    (
        State(
            soup=[[[0, 0, 9], [0, 0, 1], [0, 0, 10]]], data=[[2, 1, 0]], running=[True]
        ),
        State(
            soup=[[[0, 0, 9], [0, 0, 1], [0, 0, 10]]], data=[[0, 1, 0]], running=[True]
        ),
    ),
    (
        State(
            soup=[[[0, 0, 0], [0, 0, 1], [0, 0, 10]]], data=[[2, 1, 0]], running=[True]
        ),
        State(
            soup=[[[0, 0, 0], [0, 0, 1], [0, 0, 10]]], data=[[2, 1, 0]], running=[False]
        ),
    ),
    (
        State(
            soup=[[[0, 0, 9], [0, 0, 10], [0, 0, 0]]], data=[[1, 2, 0]], running=[True]
        ),
        State(
            soup=[[[0, 0, 9], [0, 0, 10], [0, 0, 0]]], data=[[2, 2, 0]], running=[True]
        ),
    ),
]

ids = [
    " < (single_program) ",
    " > (single_program) ",
    " { (single_program) ",
    " } (single_program) ",
    " - (single program) ",
    " + (single program) ",
    " . (single program) ",
    " , (single program) ",
    " [ (jump taken, no halt) (single program) ",
    " [ (jump taken, halted) (single program) ",
    " [ (jump not taken) (single program) ",
    " ] (jump taken, no halt) (single program) ",
    " ] (jump taken, halted) (single program) ",
    " ] (jump not taken) (single program) ",
]


@pytest.mark.parametrize(("before", "after"), tests, ids=ids)
def test(before: State, after: State):
    soup_before = torch.tensor(before.soup, dtype=torch.int32)
    data_before = torch.tensor(before.data, dtype=torch.int32)
    running_before = torch.tensor(before.running)

    step(soup_before, data_before, running_before, instruction_space_size=256)

    soup_after = torch.tensor(after.soup, dtype=torch.int32)
    data_after = torch.tensor(after.data, dtype=torch.int32)
    running_after = torch.tensor(after.running)

    assert torch.equal(soup_before, soup_after)
    assert torch.equal(data_before, data_after)
    assert torch.equal(running_before, running_after)
