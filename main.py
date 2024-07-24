import torch

from bff.interp import initialize_soup, initialize_data, step
from bff.select import select
from bff.debug import pdiff


# the number of programs to simulate in
# our turing gas.
# NOTE(Nic): num programs must be divisible by 2
num_programs = 8192
num_steps_per_epoch = 8192

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
pre_execution_programs = paired_programs.clone()


data, running = initialize_data(num_programs=num_programs // 2)

# print(paired_programs[0])

for i in range(num_steps_per_epoch):
    if i % 512 == 0:
        print(i)
    step(paired_programs, data, running, instruction_space_size=instruction_space_size)


p_idx = 0
pdiff(
    pre_execution_programs[p_idx],
    torch.zeros((3,)),
    paired_programs[p_idx],
    data[p_idx],
)

# step(soup_before, data_before, running_before, instruction_space_size=256)
