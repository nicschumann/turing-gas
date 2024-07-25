import torch
from time import perf_counter

from bff.interp import initialize_soup, initialize_data, step
from bff.select import select_pairs, return_pairs
from bff.debug import pdiff

from bff.metrics.shannon import entropy


# the number of programs to simulate in
# our turing gas.
# NOTE(Nic): num programs must be divisible by 2
num_programs = 32
num_steps_per_epoch = 8192

# the length of each program's tape
tape_length = 64

# each memory cell in the tape is a single byte, meaning we have 2^8 possible instructions
instruction_space_size = 256

# random seed for reproducibility
random_seed = 1337

device = "cpu"
# if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     device = "mps"
# elif torch.cuda.is_available():
#     torch.cuda.manual_seed(random_seed)
#     device = "cuda"

print(f"using device: {device}")


generator = torch.Generator(device=device).manual_seed(random_seed)


soup = initialize_soup(
    num_programs=num_programs,
    tape_length=tape_length,
    instruction_space_size=instruction_space_size,
    random_state=generator,
    device=device,
)

pre_soup = soup.clone()


for epoch in range(25):
    # entropy(soup, instruction_space_size=instruction_space_size)

    s_select = perf_counter()
    paired_programs, random_indices = select_pairs(
        soup, random_state=generator, device=device
    )
    e_select = perf_counter()

    s_init = perf_counter()
    data, running = initialize_data(num_programs=num_programs // 2, device=device)
    e_init = perf_counter()

    s_step = perf_counter()
    for i in range(num_steps_per_epoch):
        step(
            paired_programs,
            data,
            running,
            instruction_space_size=instruction_space_size,
            device=device,
        )
    e_step = perf_counter()

    s_ret = perf_counter()
    return_pairs(paired_programs, random_indices, soup, device=device)
    e_ret = perf_counter()

    t_step_tot = e_step - s_step
    t_step_avg = t_step_tot / num_steps_per_epoch
    prog_per_s = num_programs / t_step_avg

    t_tot = e_ret - s_select

    # perf metrics
    print(
        "\t".join(
            [
                f"e{epoch}",
                f"tot: {(1000*t_tot):.2f}ms",
                f"sel: {(1000 * (e_select - s_select)):.2f}ms",
                f"init: {(1000 * (e_init - s_init)):.2f}ms",
                f"[step avg:{(1000 * t_step_avg):.2f}ms tot: {(1000 * t_step_tot):.2f}ms p/s: {int(prog_per_s):,}]",
                f"ret: {(1000 * (e_ret - s_ret)):.2f}ms",
            ]
        )
    )


for i in range(num_programs):

    print(f"Program {i}")
    pdiff(pre_soup[i], torch.tensor([0, 0, 0]), soup[i], torch.tensor([0, 0, 0]))
    print("\n")
# p_idx = 0
# pdiff(
#     pre_execution_programs[p_idx],
#     torch.zeros((3,)),
#     paired_programs[p_idx],
#     data[p_idx],
# )

# step(soup_before, data_before, running_before, instruction_space_size=256)
