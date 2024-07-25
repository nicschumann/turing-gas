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
                dtype=torch.int32,
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
    data = torch.zeros((num_programs, 3), dtype=torch.int32, device=device)

    # NOTE(Nic): initializing it this way for debugging.
    # data = torch.tensor(
    #     [[63, 5, 1], [1, 5, 1], [63, 4, 1], [63, 0, 1]], dtype=torch.int32
    # )

    running = torch.full(
        (num_programs,), fill_value=True, dtype=torch.bool, device=device
    )

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
        active_data[h1_incr_idx, H1_IDX] %= soup.size(1)
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

            ptrs[valid_idx] = match_idx[valid_idx].to(torch.int32)

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

            ptrs[valid_idx] = match_idx[valid_idx].to(torch.int32)

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
