import torch

from bff.interp import initialize_soup, initialize_data
from bff.select import select
from bff.debug import decompile


# the number of programs to simulate in
# our turing gas.
# NOTE(Nic): num programs must be divisible by 2
num_programs = 12

# the length of each program's tape
tape_length = 64

# each memory cell in the tape is a single byte, meaning we have 2^8 possible instructions
instruction_space_size = 256

# random seed for reproducibility
random_seed = 1337

generator = torch.Generator().manual_seed(random_seed)


soup = initialize_soup(
    num_programs=num_programs,
    tape_length=tape_length,
    instruction_space_size=instruction_space_size,
    random_state=generator,
)


paired_programs, random_indices = select(soup, random_state=generator)

print(paired_programs)
print(random_indices)

data, running = initialize_data(num_programs=num_programs // 2)


for i in range(num_programs // 2):
    decompile(paired_programs[i], data[i], running[i])

# step(soup_before, data_before, running_before, instruction_space_size=256)
