import pytest
import torch
from dataclasses import dataclass

from bff.interp import step


@dataclass
class State:
    soup: list[list[list[int]]]  # NOTE(Nic): list[list[tuple[int, int, int]]]
    data: list[list[int]]  # NOTE(Nic): list[tuple[int, int, int]]
    running: list[bool]


def setup_tests():
    return [
        (
            State(
                soup=[[[0, 0, 1], [0, 0, 0], [0, 0, 0]]],
                data=[[0, 0, 0]],
                running=[True],
            ),
            State(
                soup=[[[0, 0, 1], [0, 0, 0], [0, 0, 0]]],
                data=[[1, 2, 0]],
                running=[True],
            ),
        ),
        (
            State(
                soup=[[[0, 0, 2], [0, 0, 0], [0, 0, 0]]],
                data=[[0, 0, 0]],
                running=[True],
            ),
            State(
                soup=[[[0, 0, 2], [0, 0, 0], [0, 0, 0]]],
                data=[[1, 1, 0]],
                running=[True],
            ),
        ),
        (
            State(
                soup=[[[0, 0, 3], [0, 0, 0], [0, 0, 0]]],
                data=[[0, 0, 0]],
                running=[True],
            ),
            State(
                soup=[[[0, 0, 3], [0, 0, 0], [0, 0, 0]]],
                data=[[1, 0, 2]],
                running=[True],
            ),
        ),
        (
            State(
                soup=[[[0, 0, 4], [0, 0, 0], [0, 0, 0]]],
                data=[[0, 0, 0]],
                running=[True],
            ),
            State(
                soup=[[[0, 0, 4], [0, 0, 0], [0, 0, 0]]],
                data=[[1, 0, 1]],
                running=[True],
            ),
        ),
        (
            State(
                soup=[[[0, 0, 5], [0, 0, 0], [0, 0, 0]]],
                data=[[0, 1, 0]],
                running=[True],
            ),
            State(
                soup=[[[0, 0, 5], [0, 0, 255], [0, 0, 0]]],
                data=[[1, 1, 0]],
                running=[True],
            ),
        ),
        (
            State(
                soup=[[[0, 0, 6], [0, 0, 0], [0, 0, 0]]],
                data=[[0, 1, 0]],
                running=[True],
            ),
            State(
                soup=[[[0, 0, 6], [0, 0, 1], [0, 0, 0]]],
                data=[[1, 1, 0]],
                running=[True],
            ),
        ),
        (
            State(
                soup=[[[0, 0, 7], [0, 0, 4], [0, 0, 5]]],
                data=[[0, 1, 2]],
                running=[True],
            ),
            State(
                soup=[[[0, 0, 7], [0, 0, 4], [0, 0, 4]]],
                data=[[1, 1, 2]],
                running=[True],
            ),
        ),
        (
            State(
                soup=[[[0, 0, 8], [0, 0, 4], [0, 0, 5]]],
                data=[[0, 1, 2]],
                running=[True],
            ),
            State(
                soup=[[[0, 0, 8], [0, 0, 5], [0, 0, 5]]],
                data=[[1, 1, 2]],
                running=[True],
            ),
        ),
        (
            State(
                soup=[[[0, 0, 9], [0, 0, 0], [0, 0, 10]]],
                data=[[0, 1, 0]],
                running=[True],
            ),
            State(
                soup=[[[0, 0, 9], [0, 0, 0], [0, 0, 10]]],
                data=[[2, 1, 0]],
                running=[True],
            ),
        ),
        (
            State(
                soup=[[[0, 0, 9], [0, 0, 0], [0, 0, 0]]],
                data=[[0, 1, 0]],
                running=[True],
            ),
            State(
                soup=[[[0, 0, 9], [0, 0, 0], [0, 0, 0]]],
                data=[[0, 1, 0]],
                running=[False],
            ),
        ),
        (
            State(
                soup=[[[0, 0, 9], [0, 0, 1], [0, 0, 0]]],
                data=[[0, 1, 0]],
                running=[True],
            ),
            State(
                soup=[[[0, 0, 9], [0, 0, 1], [0, 0, 0]]],
                data=[[1, 1, 0]],
                running=[True],
            ),
        ),
        (
            State(
                soup=[[[0, 0, 9], [0, 0, 1], [0, 0, 10]]],
                data=[[2, 1, 0]],
                running=[True],
            ),
            State(
                soup=[[[0, 0, 9], [0, 0, 1], [0, 0, 10]]],
                data=[[0, 1, 0]],
                running=[True],
            ),
        ),
        (
            State(
                soup=[[[0, 0, 0], [0, 0, 1], [0, 0, 10]]],
                data=[[2, 1, 0]],
                running=[True],
            ),
            State(
                soup=[[[0, 0, 0], [0, 0, 1], [0, 0, 10]]],
                data=[[2, 1, 0]],
                running=[False],
            ),
        ),
        (
            State(
                soup=[[[0, 0, 9], [0, 0, 10], [0, 0, 0]]],
                data=[[1, 2, 0]],
                running=[True],
            ),
            State(
                soup=[[[0, 0, 9], [0, 0, 10], [0, 0, 0]]],
                data=[[2, 2, 0]],
                running=[True],
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


@pytest.mark.parametrize(("before", "after"), setup_tests(), ids=ids)
def test_cpu(before: State, after: State):
    """Each of theses tests"""
    soup_before = torch.tensor(before.soup, dtype=torch.int32)
    data_before = torch.tensor(before.data, dtype=torch.int32)
    running_before = torch.tensor(before.running)

    step(
        soup_before,
        data_before,
        running_before,
        instruction_space_size=256,
        device="cpu",
    )

    soup_after = torch.tensor(after.soup, dtype=torch.int32)
    data_after = torch.tensor(after.data, dtype=torch.int32)
    running_after = torch.tensor(after.running)

    assert torch.equal(soup_before, soup_after)
    assert torch.equal(data_before, data_after)
    assert torch.equal(running_before, running_after)


def test_parallel():
    """This test runs each of the individually run tests above. However,
    it runs all 14 tests in parallel, to ensure there is no "weird crosstalk"
    or mistaken indexing between the tests; we want to ensure the parallelism
    just parallelizes, nothing else."""

    tests = setup_tests()

    soup_before = torch.tensor(
        list(map(lambda s: s[0].soup[0], tests)), dtype=torch.int32
    )
    data_before = torch.tensor(
        list(map(lambda s: s[0].data[0], tests)), dtype=torch.int32
    )
    running_before = torch.tensor(list(map(lambda s: s[0].running[0], tests)))

    step(
        soup_before,
        data_before,
        running_before,
        instruction_space_size=256,
        device="cpu",
    )

    soup_after = torch.tensor(
        list(map(lambda s: s[1].soup[0], tests)), dtype=torch.int32
    )
    data_after = torch.tensor(
        list(map(lambda s: s[1].data[0], tests)), dtype=torch.int32
    )
    running_after = torch.tensor(list(map(lambda s: s[1].running[0], tests)))

    assert torch.equal(soup_before, soup_after)
    assert torch.equal(data_before, data_after)
    assert torch.equal(running_before, running_after)
