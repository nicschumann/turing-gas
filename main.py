import torch

torch.manual_seed(1)


# the number of programs to simulate in
# our turing gas.
num_programs = 4

# the length of each program's tape
tape_length = 64

# each memory cell in the tape is a single byte, meaning we have 2^8 possible instructions
instruction_space_size = 256


# the structure of the token. We use the same structure
# as in Aguera y Arcas et al: (epoch_created, position_in_tape, character)
EPOCH_IDX = 0
POS_IDX = 1
CHAR_IDX = 2

# the structure of the data for each program
IP_IDX = 0
H0_IDX = 1
H1_IDX = 2


instructions = [
    "0",
    "<  ; decr h0",
    ">  ; incr h0",
    "{  ; decr h1",
    "}  ; incr h1",
    "-  ; decr tape[h0]",
    "+  ; incr tape[h0]",
    ".  ; mov tape[h1], tape[h0]",
    ",  ; mov tape[h0], tape[h1]",
    "[  ; jz ahead ",
    "]  ; jnz back",
]


def decompile(tokens: torch.Tensor, data: torch.Tensor, running: torch.Tensor):
    assert (
        tokens.ndim == 2 and tokens.size(-1) == 3
    ), "Expected single program with shape (length, 3)."

    ip = data[IP_IDX]
    h0 = data[H0_IDX]
    h1 = data[H1_IDX]

    print(f"            h0: {h0} | h1: {h1} | running: {running}\n")
    for i, line in enumerate(tokens):

        epoch, position, char = line
        char = char.item()
        syntax = instructions[int(char)] if char < len(instructions) else ""
        ip_flag = "ip>" if ip.item() == i else "   "
        h0_flag = "h0>" if h0.item() == i else "   "
        h1_flag = "h1>" if h1.item() == i else "   "

        print(f"{ip_flag}{h0_flag}{h1_flag}{i:4} | {epoch.item():4,} | \t\t{syntax}")


def initialize() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    soup = torch.cat(
        (
            #
            # NOTE(Nic): epoch number. all programs start at epoch 0.
            torch.full((num_programs, tape_length, 1), fill_value=0, dtype=torch.int32),
            #
            # NOTE(Nic): position. shows the original position of each token in the tape.
            torch.arange(0, tape_length)
            .repeat(num_programs)
            .reshape((num_programs, tape_length, 1)),
            #
            # NOTE(Nic): program data: sampled uniformly from all possible bytes.
            torch.randint(
                0, instruction_space_size, size=(num_programs, tape_length, 1)
            ),
            # torch.tensor(
            #     [
            #         [
            #             [9],
            #             [9],
            #             [3],
            #             [3],
            #             [3],
            #             [0],
            #             [10],
            #             [0],
            #             [0],
            #             [0],
            #         ],
            #         [
            #             [9],
            #             [10],
            #             [3],
            #             [9],
            #             [3],
            #             [1],
            #             [0],
            #             [0],
            #             [10],
            #             [0],
            #         ],
            #         [
            #             [0],
            #             [9],
            #             [0],
            #             [10],
            #             [0],
            #             [0],
            #             [10],
            #             [0],
            #             [0],
            #             [0],
            #         ],
            #         [
            #             [0],
            #             [9],
            #             [3],
            #             [9],
            #             [10],
            #             [0],
            #             [0],
            #             [10],
            #             [0],
            #             [0],
            #         ],
            #     ]
            # ),
        ),
        dim=-1,
    )

    data = torch.zeros((num_programs, 3), dtype=torch.int32)

    # NOTE(Nic): initializing it this way for debugging.
    # data = torch.tensor(
    #     [[63, 5, 1], [1, 5, 1], [63, 4, 1], [63, 0, 1]], dtype=torch.int32
    # )

    running = torch.full((num_programs,), fill_value=True, dtype=torch.bool)

    return soup, data, running


def step(soup: torch.Tensor, data: torch.Tensor, running: torch.Tensor):
    assert soup.size(0) == data.size(
        0
    ), f"expected to have the same number of programs ({soup.size(0)}) as runtime states {data.size(0)}."

    (active_programs_idx,) = torch.where(running == True)

    ips = data[active_programs_idx, IP_IDX]
    active_lines = soup[active_programs_idx, ips]

    (h0_decr_idx,) = torch.where(active_lines[:, CHAR_IDX] == 1)
    (h0_incr_idx,) = torch.where(active_lines[:, CHAR_IDX] == 2)
    (h1_decr_idx,) = torch.where(active_lines[:, CHAR_IDX] == 3)
    (h1_incr_idx,) = torch.where(active_lines[:, CHAR_IDX] == 4)
    (tape_decr_idx,) = torch.where(active_lines[:, CHAR_IDX] == 5)
    (tape_incr_idx,) = torch.where(active_lines[:, CHAR_IDX] == 6)
    (copy_to_h1_idx,) = torch.where(active_lines[:, CHAR_IDX] == 7)
    (copy_to_h0_idx,) = torch.where(active_lines[:, CHAR_IDX] == 8)
    (jump_forward_idx,) = torch.where(active_lines[:, CHAR_IDX] == 9)
    (jump_back_idx,) = torch.where(active_lines[:, CHAR_IDX] == 10)

    # print("ops:")
    # print(">\t\tp_idx:", h0_decr_idx)
    # print("<\t\tp_idx:", h0_incr_idx)
    # print("{\t\tp_idx:", h1_decr_idx)
    # print("}\t\tp_idx:", h1_incr_idx)
    # print("-\t\tp_idx:", tape_decr_idx)
    # print("+\t\tp_idx:", tape_incr_idx)
    # print(".\t\tp_idx:", copy_to_h1_idx)
    # print(",\t\tp_idx:", copy_to_h0_idx)
    # print("[\t\tp_idx:", jump_forward_idx)
    # print("]\t\tp_idx:", jump_back_idx)

    # HANDLE "<" operation
    if h0_decr_idx.size(0) > 0:
        active_data = data[active_programs_idx]
        active_data[h0_decr_idx, H0_IDX] -= 1
        active_data[h0_decr_idx, H0_IDX] %= soup.size(1)
        data[active_programs_idx] = active_data

    # HANDLE ">" operation
    if h0_incr_idx.size(0) > 0:
        active_data = data[active_programs_idx]
        active_data[h0_incr_idx, H0_IDX] += 1
        active_data[h0_incr_idx, H0_IDX] %= soup.size(1)
        data[active_programs_idx] = active_data

    # HANDLE "{" operation
    if h1_decr_idx.size(0) > 0:
        active_data = data[active_programs_idx]
        active_data[h1_decr_idx, H1_IDX] -= 1
        active_data[h1_decr_idx, H1_IDX] %= soup.size(1)
        data[active_programs_idx] = active_data

    # HANDLE "}" operation
    if h1_incr_idx.size(0) > 0:
        active_data = data[active_programs_idx]
        active_data[h1_incr_idx, H1_IDX] += 1
        active_data[h1_incr_idx, H1_IDX] %= soup.size(0)
        data[active_programs_idx] = active_data

    # HANDLE "-" operation
    if tape_decr_idx.size(0) > 0:
        # get relevant h0 values for the active programs
        h0 = data[active_programs_idx][tape_decr_idx, H0_IDX]

        p = soup[active_programs_idx]  # get all active programs

        p[tape_decr_idx, h0, CHAR_IDX] -= 1
        p[tape_decr_idx, h0, CHAR_IDX] %= instruction_space_size

        soup[active_programs_idx] = p  # write the update back into the soup.

    # HANDLE "+" operation
    if tape_incr_idx.size(0) > 0:
        # get relevant h0 values for the active programs
        h0 = data[active_programs_idx][tape_incr_idx, H0_IDX]

        p = soup[active_programs_idx]  # get all active programs

        p[tape_incr_idx, h0, CHAR_IDX] += 1
        p[tape_incr_idx, h0, CHAR_IDX] %= instruction_space_size

        soup[active_programs_idx] = p  # write the update back into the soup.

    # HANDLE "." operation
    if copy_to_h1_idx.size(0) > 0:
        # get relevant h0 values for the active programs
        h0 = data[active_programs_idx][copy_to_h1_idx, H0_IDX]
        h1 = data[active_programs_idx][copy_to_h1_idx, H1_IDX]

        p = soup[active_programs_idx]  # get all active programs

        p[copy_to_h1_idx, h1] = p[copy_to_h1_idx, h0]  # copy the entire token.

        soup[active_programs_idx] = p  # write the update back into the soup.

    # HANDLE "," operation
    if copy_to_h0_idx.size(0) > 0:
        # get relevant h0 values for the active programs
        h0 = data[active_programs_idx][copy_to_h0_idx, H0_IDX]
        h1 = data[active_programs_idx][copy_to_h0_idx, H1_IDX]

        p = soup[active_programs_idx]  # get all active programs

        p[copy_to_h0_idx, h0] = p[copy_to_h0_idx, h1]  # copy the entire token.

        soup[active_programs_idx] = p  # write the update back into the soup.

    # HANDLE "[" operation
    if jump_forward_idx.size(0) > 0:
        # get relevant h0 values for the active programs
        h0 = data[active_programs_idx][jump_forward_idx, H0_IDX]

        p = soup[active_programs_idx]  # get all active programs

        (do_jump_idx,) = torch.where(p[jump_forward_idx, h0, CHAR_IDX] == 0)

        p = p[jump_forward_idx]
        ips = data[active_programs_idx][jump_forward_idx, IP_IDX]

        if do_jump_idx.size(0) > 0:
            # we need to search forward from the current IP and find the
            # matching closing bracket, if there is one.
            ptrs = ips[do_jump_idx]

            # Find matching brackets, if there are any.
            brackets = torch.zeros_like(
                p[do_jump_idx, :, CHAR_IDX], dtype=torch.int32
            ).unsqueeze(-1)

            brackets[torch.where(p[do_jump_idx, :, CHAR_IDX] == 9)] = 1
            brackets[torch.where(p[do_jump_idx, :, CHAR_IDX] == 10)] = -1
            ip_mask = torch.arange(brackets.size(1)).unsqueeze(0) < ptrs.unsqueeze(1)
            brackets[ip_mask] = 0

            bracket_sum = torch.cumsum(brackets, dim=-2)
            bracket_sum[ip_mask] = 10

            match_vals, match_idx = torch.min(torch.absolute(bracket_sum), dim=1)
            match_vals = match_vals.squeeze(
                -1
            )  # NOTE(Nic): get rid of the leading singleton dim
            match_idx = match_idx.squeeze(-1)

            # 1) Update the instruction pointers for valid jumps, and
            (valid_idx,) = torch.where(match_vals == 0)
            (invalid_idx,) = torch.where(match_vals != 0)

            ptrs[valid_idx] = match_idx[valid_idx].to(torch.int32)

            # 2) Fever Dream of Index propagation: halt any programs that don't have a matching bracket.
            r = running[active_programs_idx]
            r_p = r[jump_back_idx]
            r_pp = r_p[do_jump_idx]
            r_pp[invalid_idx] = False
            r_p[do_jump_idx] = r_pp
            r[jump_back_idx] = r_p
            running[active_programs_idx] = r

            # Fever Dream of Index propagation: get the updates back.
            d = data[active_programs_idx]
            d_p = d[jump_forward_idx]
            d_pp = d_p[do_jump_idx]
            d_pp[valid_idx, IP_IDX] = ptrs[valid_idx] - 1
            d_p[do_jump_idx] = d_pp
            d[jump_forward_idx] = d_p
            data[active_programs_idx] = d

    # HANDLE "]" operation
    if jump_back_idx.size(0) > 0:
        # get relevant h0 values for the active programs
        h0 = data[active_programs_idx][jump_back_idx, H0_IDX]

        p = soup[active_programs_idx]  # get all active programs

        (do_jump_idx,) = torch.where(p[jump_back_idx, h0, CHAR_IDX] != 0)

        p = p[jump_back_idx]
        ips = data[active_programs_idx][jump_back_idx, IP_IDX]

        if do_jump_idx.size(0) > 0:
            # we need to search forward from the current IP and find the
            # matching closing bracket, if there is one.
            ptrs = ips[do_jump_idx]

            # Find matching brackets, if there are any.
            brackets = torch.zeros_like(
                p[do_jump_idx, :, CHAR_IDX], dtype=torch.int32
            ).unsqueeze(-1)

            brackets[torch.where(p[do_jump_idx, :, CHAR_IDX] == 9)] = 1
            brackets[torch.where(p[do_jump_idx, :, CHAR_IDX] == 10)] = -1
            ip_mask = torch.arange(brackets.size(1)).unsqueeze(0) > ptrs.unsqueeze(1)
            brackets[ip_mask] = 0

            # NOTE(Nic): these two flip operations make copies of the
            # underlying tensor, which is expensive and annoying. Replace, if slow.
            bracket_sum = torch.cumsum(torch.flip(brackets, [1]), dim=-2)
            bracket_sum[torch.flip(ip_mask, [1])] = 10

            match_vals, match_idx = torch.min(torch.absolute(bracket_sum), dim=1)

            match_vals = match_vals.squeeze(
                -1
            )  # NOTE(Nic): get rid of the leading singleton dim
            match_idx = match_idx.squeeze(-1)
            match_idx = brackets.size(1) - 1 - match_idx

            # 1) Update the instruction pointers for valid jumps, and
            (valid_idx,) = torch.where(match_vals == 0)
            (invalid_idx,) = torch.where(match_vals != 0)

            ptrs[valid_idx] = match_idx[valid_idx].to(torch.int32)

            # 2) Fever Dream of Index propagation: halt any programs that don't have a matching bracket.
            r = running[active_programs_idx]
            r_p = r[jump_back_idx]
            r_pp = r_p[do_jump_idx]
            r_pp[invalid_idx] = False
            r_p[do_jump_idx] = r_pp
            r[jump_back_idx] = r_p
            running[active_programs_idx] = r

            # Fever Dream of Index propagation: get the IP updates back.
            d = data[active_programs_idx]
            d_p = d[jump_back_idx]
            d_pp = d_p[do_jump_idx]
            d_pp[valid_idx, IP_IDX] = ptrs[valid_idx] - 1
            d_p[do_jump_idx] = d_pp
            d[jump_back_idx] = d_p
            data[active_programs_idx] = d

    # Increment all instruction pointers of valid programs.
    (active_programs_idx,) = torch.where(running == True)
    data[active_programs_idx, IP_IDX] += 1
    running[torch.where(data[:, IP_IDX] >= soup.size(1))] = (
        False  # NOTE(Nic): halt programs with OOB IPs
    )


soup, data, running = initialize()

print("\npre-step soup state:\n")
for i, program in enumerate(soup):
    decompile(program, data[i], running[i])
    print("\n")

step(soup, data, running)

print("\npost-step soup state:\n")
for i, program in enumerate(soup):
    decompile(program, data[i], running[i])
    print("\n")


# step(soup, data, running)

# print("\nsoup state:")
# print(soup)

# print("\ndata state:")
# print(data)

# print("\nrunning state:")
# print(running)
