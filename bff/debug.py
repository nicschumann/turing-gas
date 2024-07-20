import torch

from .interp import IP_IDX, H0_IDX, H1_IDX


instructions = [
    "0",
    "<  ; decr h0",
    ">  ; incr h0",
    "{  ; decr h1",
    "}  ; incr h1",
    "-  ; decr tape[h0]",
    "+  ; incr tape[h0]",
    ".  ; mov tape[h1] <- tape[h0]",
    ",  ; mov tape[h0] <- tape[h1]",
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
