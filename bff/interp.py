import torch
from torch import Tensor
from jaxtyping import Int, Bool

# the structure of the token. We use the same structure
# as in Aguera y Arcas et al: (epoch_created, position_in_tape, character)
EPOCH_IDX = 0
POS_IDX = 1
CHAR_IDX = 2

# the structure of the data for each program
IP_IDX = 0
H0_IDX = 1
H1_IDX = 2


def initialize_soup(
    *,
    num_programs: int,
    tape_length: int,
    instruction_space_size: int,
    random_state: torch.Generator | None,
    device: str,
) -> Int[Tensor, "programs length data"]:
    soup = torch.cat(
        (
            #
            # NOTE(Nic): epoch number. all programs start at epoch 0.
            torch.full(
                (num_programs, tape_length, 1),
                fill_value=0,
                device=device,
            ),
            #
            # NOTE(Nic): position. shows the original program index that each token belonged to at initialization.
            torch.arange(0, num_programs, device=device)
            .repeat(tape_length)
            .reshape(tape_length, num_programs)
            .transpose(0, 1)
            .unsqueeze(-1),
            #
            # NOTE(Nic): program data: sampled uniformly from all possible bytes.
            torch.randint(
                0,
                instruction_space_size,
                size=(num_programs, tape_length, 1),
                generator=random_state,
                device=device,
            ),
        ),
        dim=-1,
    )

    return soup


def initialize_data(
    *, num_programs: int, device: str
) -> tuple[Int[Tensor, "programs data"], Bool[Tensor, "programs"]]:
    data = torch.zeros((num_programs, 3), dtype=torch.long, device=device)
    running = torch.full((num_programs,), fill_value=True, device=device)

    return data, running


def step(
    soup: torch.Tensor,
    data: torch.Tensor,
    running: torch.Tensor,
    *,
    instruction_space_size: int,
    device: str,
):
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
        active_data[h1_decr_idx, H0_IDX].sub_(1).fmod_(soup.shape[1])
        data[active_programs_idx] = active_data

    # HANDLE ">" operation
    if h0_incr_idx.size(0) > 0:
        active_data = data[active_programs_idx]
        active_data[h1_decr_idx, H0_IDX].add_(1).fmod_(soup.shape[1])
        data[active_programs_idx] = active_data

    # HANDLE "{" operation
    if h1_decr_idx.size(0) > 0:
        active_data = data[active_programs_idx]
        active_data[h1_decr_idx, H1_IDX].sub_(1).fmod_(soup.shape[1])
        data[active_programs_idx] = active_data

    # HANDLE "}" operation
    if h1_incr_idx.size(0) > 0:
        active_data = data[active_programs_idx]
        active_data[h1_decr_idx, H1_IDX].add_(1).fmod_(soup.shape[1])
        data[active_programs_idx] = active_data

    # HANDLE "-" operation
    if tape_decr_idx.size(0) > 0:
        # get relevant h0 values for the active programs
        h0 = data[active_programs_idx][tape_decr_idx, H0_IDX]

        p = soup[active_programs_idx]  # get all active programs

        p[tape_decr_idx, h0, CHAR_IDX].sub_(1).fmod_(instruction_space_size)

        soup[active_programs_idx] = p  # write the update back into the soup.

    # HANDLE "+" operation
    if tape_incr_idx.size(0) > 0:
        # get relevant h0 values for the active programs
        h0 = data[active_programs_idx][tape_incr_idx, H0_IDX]

        p = soup[active_programs_idx]  # get all active programs

        p[tape_incr_idx, h0, CHAR_IDX].add_(1).fmod_(instruction_space_size)

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
                p[do_jump_idx, :, CHAR_IDX], dtype=torch.int32, device=device
            ).unsqueeze(-1)

            brackets[torch.where(p[do_jump_idx, :, CHAR_IDX] == 9)] = 1
            brackets[torch.where(p[do_jump_idx, :, CHAR_IDX] == 10)] = -1
            ip_mask = torch.arange(brackets.size(1), device=device).unsqueeze(
                0
            ) < ptrs.unsqueeze(1)
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

            ptrs[valid_idx] = match_idx[valid_idx]

            # 2) Fever Dream of Index propagation: halt any programs that don't have a matching bracket.
            if invalid_idx.size(0) > 0:
                r = running[active_programs_idx]
                r_p = r[jump_forward_idx]
                r_pp = r_p[do_jump_idx]
                r_pp[invalid_idx] = False
                r_p[do_jump_idx] = r_pp
                r[jump_forward_idx] = r_p
                running[active_programs_idx] = r

            # Fever Dream of Index propagation: get the updates back.
            if valid_idx.size(0) > 0:
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
                p[do_jump_idx, :, CHAR_IDX], dtype=torch.int32, device=device
            ).unsqueeze(-1)

            brackets[torch.where(p[do_jump_idx, :, CHAR_IDX] == 9)] = 1
            brackets[torch.where(p[do_jump_idx, :, CHAR_IDX] == 10)] = -1
            ip_mask = torch.arange(brackets.size(1), device=device).unsqueeze(
                0
            ) > ptrs.unsqueeze(1)
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

            ptrs[valid_idx] = match_idx[valid_idx]

            # 2) Fever Dream of Index propagation: halt any programs that don't have a matching bracket.
            if invalid_idx.size(0) > 0:
                r = running[active_programs_idx]
                r_p = r[jump_back_idx]
                r_pp = r_p[do_jump_idx]
                r_pp[invalid_idx] = False
                r_p[do_jump_idx] = r_pp
                r[jump_back_idx] = r_p
                running[active_programs_idx] = r

            # Fever Dream of Index propagation: get the IP updates back.
            if valid_idx.size(0) > 0:
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


def step2(
    soup: torch.Tensor,
    data: torch.Tensor,
    running: torch.Tensor,
    *,
    instruction_space_size: int,
    device: str,
):
    num_programs, tape_length, _ = soup.shape

    r = torch.arange(num_programs, device=soup.device)
    d = data[:, IP_IDX]

    instructions = soup[
        r,
        d,
    ]

    # Update state based on instructions (only for running soup)
    update_state(
        soup,
        data,
        instructions,
        running,
        instruction_space_size=instruction_space_size,
    )

    # Update soup if necessary (only for running programs)
    update_programs(soup, data, running, instruction_space_size=instruction_space_size)

    # Increment instruction pointer (only for running soup)
    data[:, IP_IDX].add_(running.long()).fmod_(tape_length)


def update_state(
    soup: torch.Tensor,
    data: torch.Tensor,
    instructions: torch.Tensor,
    running: torch.Tensor,
    *,
    instruction_space_size: int,
):
    num_programs, tape_length, _ = soup.shape

    mask_1 = running & (instructions[:, CHAR_IDX] == 1)  # decr h0
    data[mask_1, H0_IDX] = (data[mask_1, H0_IDX] - 1) % soup.size(1)

    mask_2 = running & (instructions[:, CHAR_IDX] == 2)  # incr h0
    data[mask_2, H0_IDX] = (data[mask_2, H0_IDX] + 1) % soup.size(1)

    mask_3 = running & (instructions[:, CHAR_IDX] == 3)  # decr h1
    data[mask_3, H1_IDX] = (data[mask_3, H1_IDX] - 1) % soup.size(1)

    mask_4 = running & (instructions[:, CHAR_IDX] == 4)  # incr h1
    data[mask_4, H1_IDX] = (data[mask_4, H1_IDX] + 1) % soup.size(1)

    handle_jumps(
        soup,
        data,
        instructions,
        running,
        instruction_space_size=instruction_space_size,
    )


def handle_jumps(
    soup: torch.Tensor,
    data: torch.Tensor,
    instructions: torch.Tensor,
    running: torch.Tensor,
    *,
    instruction_space_size: int,
):

    num_programs, tape_length, _ = soup.shape
    mask_open = (
        running
        & (instructions[:, CHAR_IDX] == 9)
        & (
            soup[
                torch.arange(num_programs),
                data[:, H0_IDX],
                torch.full((num_programs,), fill_value=CHAR_IDX),
            ]
            == 0
        )
    )

    if mask_open.any():
        new_ip, success = find_matching_close(soup[mask_open], data[mask_open, IP_IDX])
        data[mask_open, IP_IDX] = new_ip
        running[mask_open] = success

    mask_close = (
        running
        & (instructions[:, CHAR_IDX] == 10)
        & (
            soup[
                torch.arange(num_programs),
                data[:, H0_IDX],
                torch.full((num_programs,), fill_value=CHAR_IDX),
            ]
            != 0
        )
    )
    if mask_close.any():
        new_ip, success = find_matching_close(
            soup[mask_close], data[mask_close, IP_IDX]
        )
        data[mask_close, IP_IDX] = new_ip
        running[mask_close] = success


def find_matching_brackets(
    programs: torch.Tensor, start_positions: torch.Tensor, direction: str
):
    batch_size, program_length, _ = programs.shape
    positions = start_positions.unsqueeze(1).expand(-1, program_length)
    indices = (
        torch.arange(program_length, device=programs.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    if direction == "forward":
        indices = (start_positions.unsqueeze(1) + indices) % program_length
        open_bracket, close_bracket = 9, 10
    else:  # backward
        indices = (start_positions.unsqueeze(1) - indices) % program_length
        open_bracket, close_bracket = 10, 9

    programs_expanded = programs[:, :, CHAR_IDX].gather(1, indices)

    open_mask = programs_expanded == open_bracket
    close_mask = programs_expanded == close_bracket

    nesting_level = torch.cumsum(open_mask.long() - close_mask.long(), dim=1)

    matching_positions = (nesting_level == 0) & (
        indices != start_positions.unsqueeze(1)
    )
    first_match = matching_positions.long().argmax(dim=1)

    success = matching_positions.any(dim=1)
    result_positions = indices.gather(1, first_match.unsqueeze(1)).squeeze(1)
    result_positions[
        success
    ] -= 1  # NOTE(Nic): decrement IP for successful jumps only (it will be incremented)

    return result_positions, success


def find_matching_close(programs, start_positions):
    return find_matching_brackets(programs, start_positions, "forward")


def find_matching_open(programs, start_positions):
    return find_matching_brackets(programs, start_positions, "backward")


def update_programs(
    soup: torch.Tensor,
    data: torch.Tensor,
    running: torch.Tensor,
    *,
    instruction_space_size: int,
):
    num_programs, program_length, _ = soup.shape
    char_idx_range = torch.full(
        (num_programs,), fill_value=CHAR_IDX, device=soup.device
    )

    # Only update running programs
    mask = running

    # Decrement program[h0]
    mask_5 = mask & (
        soup[
            torch.arange(num_programs),
            data[:, IP_IDX],
            char_idx_range,
        ]
        == 5
    )

    soup[mask_5, data[mask_5, H0_IDX], CHAR_IDX] = (
        soup[mask_5, data[mask_5, H0_IDX], CHAR_IDX] - 1
    ) % instruction_space_size

    # Increment program[h0]
    mask_6 = mask & (
        soup[torch.arange(num_programs), data[:, IP_IDX], char_idx_range] == 6
    )
    soup[mask_6, data[mask_6, H0_IDX], CHAR_IDX] = (
        soup[mask_6, data[mask_6, H0_IDX], CHAR_IDX] + 1
    ) % instruction_space_size

    # Move program[h0] to program[h1]
    mask_7 = mask & (
        soup[torch.arange(num_programs), data[:, IP_IDX], char_idx_range] == 7
    )
    soup[mask_7, data[mask_7, H1_IDX]] = soup[mask_7, data[mask_7, H0_IDX]]

    # Move program[h1] to program[h0]
    mask_8 = mask & (
        soup[torch.arange(num_programs), data[:, IP_IDX], char_idx_range] == 8
    )
    soup[mask_8, data[mask_8, H0_IDX]] = soup[mask_8, data[mask_8, H1_IDX]]
