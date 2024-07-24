import torch
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.padding import Padding

from .interp import IP_IDX, H0_IDX, H1_IDX

# NOTE(Nic): some of these are raw strings, because rich uses \[ to escape stuff, but it's not a valid escape sequence
# as of 3.12, python reports a SyntaxWarning with invalid escape sequences. We're don't need
# to escape anything, so we just use raw strings here.
instructions = [
    "[bold]0x0[/bold]",
    "[bold]decr[/bold] [green1]h0[/green1]",
    "[bold]incr[/bold] [green1]h0[/green1]",
    "[bold]decr[/bold] [dodger_blue2]h1[/dodger_blue2]",
    "[bold]incr[/bold] [dodger_blue2]h1[/dodger_blue2]",
    r"[bold]decr[/bold] tape[bright_black]\[[/bright_black][green1]h0[/green1][bright_black]][/bright_black]",
    r"[bold]incr[/bold] tape[bright_black]\[[/bright_black][green1]h0[/green1][bright_black]][/bright_black]",
    r"[bold magenta]mov[/bold magenta] tape[bright_black]\[[/bright_black][dodger_blue2]h1[/dodger_blue2][bright_black]][/bright_black] [bright_black]<-[/bright_black] tape[bright_black]\[[/bright_black][green1]h0[/green1][bright_black]][/bright_black]",
    r"[bold magenta]mov[/bold magenta] tape[bright_black]\[[/bright_black][green1]h0[/green1][bright_black]][/bright_black] [bright_black]<-[/bright_black] tape[bright_black]\[[/bright_black][dodger_blue2]h1[/dodger_blue2][bright_black]][/bright_black]",
    r"[bold yellow]\[[/bold yellow] [dark_green]; go to next matching ] [/dark_green]",
    r"[bold yellow]][/bold yellow] [dark_green]; go to prev matching \[ [/dark_green]",
]


def pdiff(
    p1: torch.Tensor,
    d1: torch.Tensor,
    p2: torch.Tensor,
    d2: torch.Tensor,
    *,
    truncate_noops: bool = True,
):
    assert p1.size(1) == p2.size(1), f"Can't diff programs of different lengths!"

    table = Table(
        show_header=True,
        pad_edge=True,
        box=box.SIMPLE_HEAD,
        border_style="bright_black",
    )
    table.add_column("line", width=12)
    table.add_column("p1", width=3)
    table.add_column("ip", width=2)
    table.add_column("h0", width=2)
    table.add_column("h1", width=2)
    table.add_column("", no_wrap=True, min_width=35)
    table.add_column("p2", width=3)
    table.add_column("ip", width=2)
    table.add_column("h0", width=2)
    table.add_column("h1", width=2)
    table.add_column("", no_wrap=True, min_width=35)

    prev_was_instruction = True

    p1_ip = d1[IP_IDX]
    p1_h0 = d1[H0_IDX]
    p1_h1 = d1[H1_IDX]

    p2_ip = d2[IP_IDX]
    p2_h0 = d2[H0_IDX]
    p2_h1 = d2[H1_IDX]

    for i, line in enumerate(p1):
        p1_e, p1_pnum, p1_instr = p1[i]
        p2_e, p2_pnum, p2_instr = p2[i]

        p1_ip_here = i == p1_ip.item()
        p2_ip_here = i == p2_ip.item()

        p1_h0_here = i == p1_h0.item()
        p2_h0_here = i == p2_h0.item()

        p1_h1_here = i == p1_h1.item()
        p2_h1_here = i == p2_h1.item()

        color = "bright_black" if p1_instr == p2_instr else "bold red"

        p1_instr_text = (
            instructions[int(p1_instr.item())]
            if p1_instr < len(instructions)
            else f"[bright_black]{hex(p1_instr)}[/bright_black]"
        )
        p2_instr_text = (
            instructions[int(p2_instr.item())]
            if p2_instr < len(instructions)
            else f"[bright_black]{hex(p2_instr)}[/bright_black]"
        )

        if (
            p1_instr.item() < len(instructions)
            or p2_instr.item() < len(instructions)
            or p1_instr != p2_instr
            or p1_ip_here
            or p2_ip_here
            or p1_h0_here
            or p2_h0_here
            or p1_h1_here
            or p2_h1_here
            or not truncate_noops
        ):
            table.add_row(
                f"[{color}]{i:3d}[/{color}]",
                f"[{color}]{p1_e.item()}/{p1_pnum.item()}[/{color}]",
                Text(
                    "ip" if p1_ip_here else "",
                    style="bold yellow" if p1_ip_here else "",
                ),
                Text(
                    "h0" if p1_ip_here else "",
                    style="bold green1" if p1_h0_here else "",
                ),
                Text(
                    "h1" if p1_ip_here else "",
                    style="bold dodger_blue2" if p1_h1_here else "",
                ),
                p1_instr_text,
                f"[{color}]{p1_e.item()}/{p1_pnum.item()}[/{color}]",
                Text(
                    "ip" if p2_ip_here else "",
                    style="bold yellow" if p2_ip_here else "",
                ),
                Text(
                    "h0" if p2_h0_here else "",
                    style="bold green1" if p2_h0_here else "",
                ),
                Text(
                    "h1" if p2_h1_here else "",
                    style="bold dodger_blue2" if p2_h1_here else "",
                ),
                p2_instr_text,
            )
            prev_was_instruction = True

        elif truncate_noops and prev_was_instruction:
            table.add_row(
                "[bright_black]  ⋮[/bright_black]",
                "",
                "",
                "",
                "",
                "[bright_black]⋮[/bright_black]",
                "",
                "",
                "",
                "",
                "[bright_black]⋮[/bright_black]",
            )
            prev_was_instruction = False

    console = Console()
    console.print(table)
