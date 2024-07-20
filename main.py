import torch

torch.manual_seed(1)

from bff.interp import initialize, step
from bff.debug import decompile


# the number of programs to simulate in
# our turing gas.
num_programs = 8

# the length of each program's tape
tape_length = 64

# each memory cell in the tape is a single byte, meaning we have 2^8 possible instructions
instruction_space_size = 256


soup_before = torch.tensor([[[0, 0, 9], [0, 0, 0], [0, 0, 0]]], dtype=torch.int32)
data_before = torch.tensor([[0, 1, 0]], dtype=torch.int32)
running_before = torch.tensor([True])

step(soup_before, data_before, running_before, instruction_space_size=256)
