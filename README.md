# Exploring Church/Curry/Turing Gasses

This repo is a reproduction and extension of the work in Agüera y Arcas et al's July 2024 paper "Computational Life: How Well-formed Self-replicating Programs Emerge from Simple Interaction". I've got a few goals with this effort.

1. Replicate the effort described in [the paper](https://arxiv.org/pdf/2406.19108) and in [their repo](https://github.com/paradigms-of-intelligence/cubff) in vanilla pytorch, which I find easier and read and document than their c++/cuda implementation. This also finds the benefit of being portable to other backends beyond cuda (as a mac laptop user, I want to run sims on the `mips` backend on my mac, for example).

1. Enrich the language of metrics and visualizations that the authors introduce for analyzing the structure and dynamics of the gas. I'm interested in more detailed tracing of the evolution and heredity of programs in the gas. To do this, I want to develop more robust tracking systems that map the gas, making it easy to compare programs in the gas according to the "shape" of their execution traces. Additionally, I want to map and compare various phase spaces of these execution traces, as a means to understand similarity (and property heredity) between programs. What are these phase spaces? Eventually, read my [write up](https://docs.google.com/document/d/1KRbq_mHJJE5VDQ6Q9jO7iVicyvJeINX3AaedCgcEXxk/edit?usp=sharing) to solve this mystery.

As an accompanyment to this repository, I'm also interested in producing a detailed write-up, which goes into more implementation detail than "Computational Life". [I'm working on that in this document](https://docs.google.com/document/d/1KRbq_mHJJE5VDQ6Q9jO7iVicyvJeINX3AaedCgcEXxk/edit?usp=sharing).

## Setup

This repo is intended to be dependency-minimal. The only dependency required to run the core simulation is `pytorch` (although we also use `jaxtyping` to type our tensors). Visualization requires additional layers to make your terminal prettier.

Install requirements in a virtual environment through `pip`.

```sh
> python3 -m venv .env  # or whatever you want to name your venv
> source .env/bin/activate
> pip install -r requirements.txt
```

This should get you all of the requirements you need.

## Running

> TODO(**Nic**): Once there's a proper entrypoint that spins up a simulation, add a section here to specify how to do that.

Currently, `main.py` is just an environment I'm using for interactive testing of the BFF functions.

## Testing

Tests are handled throigh `pytest`. Currently, there are tests for each operation of the `bff` virtual machine, as well as 6 tests for the jump instructions `[` and `]`, which are by far the most complex. Each of these tests runs on a single, very simple, test program (albeit using the same vectorized `step` function that runs program ensembles). Next, I plan to add testing for a program enemble that tests each operation in parallel, to ensure that there are no weird interactions between programs as a product of vectorization bugs or bad vector indexing in my implementation.

To run tests:

```sh
pytest -v  # verbose will show what each test actually tests.
```

## Future Interests

**Other Rewriting Systems**. I'm interested in extending the Turing-machine-based explorations Agüera y Arcas et al engage with to Turing-equivalent rewriting systems; simpler versions of those described in Walter Fontana's [1990 paper](https://sfi-edu.s3.amazonaws.com/sfi-edu/production/uploads/sfi-com/dev/uploads/filer/9e/f0/9ef0cbc7-8fe9-4fea-816d-6ce4a117e248/90-011.pdf) "Algorithmic Chemistry: A Model for Functional Self-Organization". The rewriting system Fontana studies is quite complex; I'm interested in much more austere systems (the [SKI Calculus](https://people.cs.uchicago.edu/~odonnell/Teacher/Lectures/Formal_Organization_of_Knowledge/Examples/combinator_calculus/https://people.cs.uchicago.edu/~odonnell/Teacher/Lectures/Formal_Organization_of_Knowledge/Examples/combinator_calculus/), or Gödel renumberings of it like [Jot](https://web.archive.org/web/20160823182917/http://semarch.linguistics.fas.nyu.edu/barker/Iota/), for example). As an aside, it will also be fun to try and vectorize these systems to be fast on `mips` and `cuda`.

## References

**Agüera y Arcas et al**. "Computational Life: How Well-formed Self-replicating Programs Emerge from Simple Interaction." June 27th, 2024. arXiv preprint [arXiv:2406.19108](https://arxiv.org/pdf/2406.19108)

**Fontana, Walter**. "Algorithmic Chemistry: A Model for Functional Self-Organization". June 9th, 1990. SFI Working Papers: [1990--011](https://sfi-edu.s3.amazonaws.com/sfi-edu/production/uploads/sfi-com/dev/uploads/filer/9e/f0/9ef0cbc7-8fe9-4fea-816d-6ce4a117e248/90-011.pdf).
