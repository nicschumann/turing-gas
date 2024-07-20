import torch

from bff.interp import step


def test_langlebracket__single_program():
    """Tests the "<" instruction, which decrements the value of the h0 register"""
    soup_before = torch.tensor([[[0, 0, 1], [0, 0, 0], [0, 0, 0]]])
    data_before = torch.tensor([[0, 0, 0]])
    running_before = torch.tensor([True])

    step(soup_before, data_before, running_before, instruction_space_size=256)

    soup_after = torch.tensor([[[0, 0, 1], [0, 0, 0], [0, 0, 0]]])
    data_after = torch.tensor([[1, 2, 0]])
    running_after = torch.tensor([True])

    assert torch.equal(soup_before, soup_after)
    assert torch.equal(data_before, data_after)
    assert torch.equal(running_before, running_after)


def test_ranglebracket__single_program():
    """Tests the ">" instruction, which increments the value of the h0 register"""
    soup_before = torch.tensor([[[0, 0, 2], [0, 0, 0], [0, 0, 0]]])
    data_before = torch.tensor([[0, 0, 0]])
    running_before = torch.tensor([True])

    step(soup_before, data_before, running_before, instruction_space_size=256)

    soup_after = torch.tensor([[[0, 0, 2], [0, 0, 0], [0, 0, 0]]])
    data_after = torch.tensor([[1, 1, 0]])
    running_after = torch.tensor([True])

    assert torch.equal(soup_before, soup_after)
    assert torch.equal(data_before, data_after)
    assert torch.equal(running_before, running_after)


def test_lbrace__single_program():
    """Tests the "{" instruction, which decrements the value of the h1 register"""
    soup_before = torch.tensor([[[0, 0, 3], [0, 0, 0], [0, 0, 0]]])
    data_before = torch.tensor([[0, 0, 0]])
    running_before = torch.tensor([True])

    step(soup_before, data_before, running_before, instruction_space_size=256)

    soup_after = torch.tensor([[[0, 0, 3], [0, 0, 0], [0, 0, 0]]])
    data_after = torch.tensor([[1, 0, 2]])
    running_after = torch.tensor([True])

    assert torch.equal(soup_before, soup_after)
    assert torch.equal(data_before, data_after)
    assert torch.equal(running_before, running_after)


def test_rbrace__single_program():
    """Tests the "}" instruction, which increments the value of the h1 register"""
    soup_before = torch.tensor([[[0, 0, 4], [0, 0, 0], [0, 0, 0]]])
    data_before = torch.tensor([[0, 0, 0]])
    running_before = torch.tensor([True])

    step(soup_before, data_before, running_before, instruction_space_size=256)

    soup_after = torch.tensor([[[0, 0, 4], [0, 0, 0], [0, 0, 0]]])
    data_after = torch.tensor([[1, 0, 1]])
    running_after = torch.tensor([True])

    assert torch.equal(soup_before, soup_after)
    assert torch.equal(data_before, data_after)
    assert torch.equal(running_before, running_after)


def test_minus__single_program():
    """Tests the "-" instruction, which decrements the value of the tape cell pointed to by h0"""
    soup_before = torch.tensor([[[0, 0, 5], [0, 0, 0], [0, 0, 0]]])
    data_before = torch.tensor([[0, 1, 0]])
    running_before = torch.tensor([True])

    step(soup_before, data_before, running_before, instruction_space_size=256)

    soup_after = torch.tensor([[[0, 0, 5], [0, 0, 255], [0, 0, 0]]])
    data_after = torch.tensor([[1, 1, 0]])
    running_after = torch.tensor([True])

    assert torch.equal(soup_before, soup_after)
    assert torch.equal(data_before, data_after)
    assert torch.equal(running_before, running_after)


def test_plus__single_program():
    """Tests the "+" instruction, which increments the value of the tape cell pointed to by h0"""
    soup_before = torch.tensor([[[0, 0, 6], [0, 0, 0], [0, 0, 0]]])
    data_before = torch.tensor([[0, 2, 0]])
    running_before = torch.tensor([True])

    step(soup_before, data_before, running_before, instruction_space_size=256)

    soup_after = torch.tensor([[[0, 0, 6], [0, 0, 0], [0, 0, 1]]])
    data_after = torch.tensor([[1, 2, 0]])
    running_after = torch.tensor([True])

    assert torch.equal(soup_before, soup_after)
    assert torch.equal(data_before, data_after)
    assert torch.equal(running_before, running_after)


def test_period__single_program():
    """Tests the "." instruction, which copies the value of tape at h0 into tape at h1"""
    soup_before = torch.tensor([[[0, 0, 7], [0, 0, 4], [0, 0, 5]]])
    data_before = torch.tensor([[0, 1, 2]])
    running_before = torch.tensor([True])

    step(soup_before, data_before, running_before, instruction_space_size=256)

    soup_after = torch.tensor([[[0, 0, 7], [0, 0, 4], [0, 0, 4]]])
    data_after = torch.tensor([[1, 1, 2]])
    running_after = torch.tensor([True])

    assert torch.equal(soup_before, soup_after)
    assert torch.equal(data_before, data_after)
    assert torch.equal(running_before, running_after)


def test_comma__single_program():
    """Tests the "," instruction, which copies the value of tape at h1 into tape at h0"""
    soup_before = torch.tensor([[[0, 0, 8], [0, 0, 4], [0, 0, 5]]])
    data_before = torch.tensor([[0, 1, 2]])
    running_before = torch.tensor([True])

    step(soup_before, data_before, running_before, instruction_space_size=256)

    soup_after = torch.tensor([[[0, 0, 8], [0, 0, 5], [0, 0, 5]]])
    data_after = torch.tensor([[1, 1, 2]])
    running_after = torch.tensor([True])

    assert torch.equal(soup_before, soup_after)
    assert torch.equal(data_before, data_after)
    assert torch.equal(running_before, running_after)


def test_lbracket__jump_taken_no_halt__single_program():
    """Tests the "[" instruction, assuming a forward jump is correctly taken."""
    soup_before = torch.tensor([[[0, 0, 9], [0, 0, 0], [0, 0, 10]]], dtype=torch.int32)
    data_before = torch.tensor([[0, 1, 0]], dtype=torch.int32)
    running_before = torch.tensor([True])

    step(soup_before, data_before, running_before, instruction_space_size=256)

    soup_after = torch.tensor([[[0, 0, 9], [0, 0, 0], [0, 0, 10]]], dtype=torch.int32)
    data_after = torch.tensor([[2, 1, 0]], dtype=torch.int32)
    running_after = torch.tensor([True])

    assert torch.equal(soup_before, soup_after)
    assert torch.equal(data_before, data_after)
    assert torch.equal(running_before, running_after)


def test_lbracket__jump_taken_halt__single_program():
    """Tests the "[" instruction, assuming a forward jump is taken, but the program is malformed"""
    soup_before = torch.tensor([[[0, 0, 9], [0, 0, 0], [0, 0, 0]]], dtype=torch.int32)
    data_before = torch.tensor([[0, 1, 0]], dtype=torch.int32)
    running_before = torch.tensor([True])

    step(soup_before, data_before, running_before, instruction_space_size=256)

    soup_after = torch.tensor([[[0, 0, 9], [0, 0, 0], [0, 0, 0]]], dtype=torch.int32)
    data_after = torch.tensor([[0, 1, 0]], dtype=torch.int32)
    running_after = torch.tensor([False])

    assert torch.equal(soup_before, soup_after)
    assert torch.equal(data_before, data_after)
    assert torch.equal(running_before, running_after)


def test_lbracket__jump_not_taken__single_program():
    """Tests the "[" instruction, assuming a forward jump is not taken."""
    soup_before = torch.tensor([[[0, 0, 9], [0, 0, 1], [0, 0, 0]]], dtype=torch.int32)
    data_before = torch.tensor([[0, 1, 0]], dtype=torch.int32)
    running_before = torch.tensor([True])

    step(soup_before, data_before, running_before, instruction_space_size=256)

    soup_after = torch.tensor([[[0, 0, 9], [0, 0, 1], [0, 0, 0]]], dtype=torch.int32)
    data_after = torch.tensor([[1, 1, 0]], dtype=torch.int32)
    running_after = torch.tensor([True])

    assert torch.equal(soup_before, soup_after)
    assert torch.equal(data_before, data_after)
    assert torch.equal(running_before, running_after)


def test_rbracket__jump_taken_no_halt__single_program():
    """Tests the "]" instruction, assuming a backward jump is correctly taken."""
    soup_before = torch.tensor([[[0, 0, 9], [0, 0, 1], [0, 0, 10]]], dtype=torch.int32)
    data_before = torch.tensor([[2, 1, 0]], dtype=torch.int32)
    running_before = torch.tensor([True])

    step(soup_before, data_before, running_before, instruction_space_size=256)

    soup_after = torch.tensor([[[0, 0, 9], [0, 0, 1], [0, 0, 10]]], dtype=torch.int32)
    data_after = torch.tensor([[0, 1, 0]], dtype=torch.int32)
    running_after = torch.tensor([True])

    assert torch.equal(soup_before, soup_after)
    assert torch.equal(data_before, data_after)
    assert torch.equal(running_before, running_after)


def test_rbracket__jump_taken_halt__single_program():
    """Tests the "]" instruction, assuming a backward jump is taken, but the program is malformed"""
    soup_before = torch.tensor([[[0, 0, 3], [0, 0, 1], [0, 0, 10]]], dtype=torch.int32)
    data_before = torch.tensor([[2, 1, 0]], dtype=torch.int32)
    running_before = torch.tensor([True])

    step(soup_before, data_before, running_before, instruction_space_size=256)

    soup_after = torch.tensor([[[0, 0, 3], [0, 0, 1], [0, 0, 10]]], dtype=torch.int32)
    data_after = torch.tensor([[2, 1, 0]], dtype=torch.int32)
    running_after = torch.tensor([False])

    assert torch.equal(soup_before, soup_after)
    assert torch.equal(data_before, data_after)
    assert torch.equal(running_before, running_after)


def test_rbracket__jump_not_taken__single_program():
    """Tests the "]" instruction, assuming a backward jump is taken, but the program is malformed"""
    soup_before = torch.tensor([[[0, 0, 0], [0, 0, 10], [0, 0, 0]]], dtype=torch.int32)
    data_before = torch.tensor([[1, 0, 0]], dtype=torch.int32)
    running_before = torch.tensor([True])

    step(soup_before, data_before, running_before, instruction_space_size=256)

    soup_after = torch.tensor([[[0, 0, 0], [0, 0, 10], [0, 0, 0]]], dtype=torch.int32)
    data_after = torch.tensor([[2, 0, 0]], dtype=torch.int32)
    running_after = torch.tensor([True])

    assert torch.equal(soup_before, soup_after)
    assert torch.equal(data_before, data_after)
    assert torch.equal(running_before, running_after)
