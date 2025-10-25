# Branching process population dynamics vs SDE model
We have a population that branches according to a 2nd degree order polynomial, which
means the rate per individual at which individuals die increases with the population
size and the individual birth rate decreases with population size.
We also include migration into the population as a Poisson variable with a given rate.

We create a SDE model, whose parameters can be derived from first principles as an
approximation to the branching process. We have shown in our paper that the first two
moment equations for the two distributions are the same, which means they have the
same Gaussian closure, at the very least.

In this package, we create analysis tools to sample the population distribution
at given times in both the branching process and its SDE approximation.
We also use these tools to create an analysis of how the distributions differ in
certain cases using a KS test.

## How to install this package
This is mostly a rust package. We will hopefully supply a complete binary version,
but you can always simply use `cargo` to `cargo build` and run the application
yourself.

The python module is used for comparisons with other codes and for a reference solution,
and you probably don't need it yourself, unless you're trying to review our use of this reference.
One can install this package with `pip install -e .`, and the test requirements with `pip install -r test_requirements.txt`.
One can then test the code by running `pytest -v` in the `pytests` folder.


## How to use this code
We mostly expect researchers to want one of two things: Either they want to generate 
paths of the population distribution, or they want to sample from the asymptotic distribution.
We provide tools for both cases in both models.

Examples are given in the `runs` directory, and the `scripts` directory includes some scripts that show how this code can be run.
The examples in the `runs` folder show the inputs we use in an upcoming paper which we hope to publish soon.

## Project Structure
Under the `src` directory you will find our source code, written in Rust. 

Under the `runs` directory you will find examples for how to use the code. These appear as scripts.

Under the `tests` directory you will find our unit tests.
We could have more of these, and if we get more researchers interested in this code this would be higher own our to-do list.
You can consider contributing tests yourself, actually. It's a good place for new contributers to start.

## Implementation
Our implementation is currently done in Rust. It's faster this way than with Python.


# Meaning of branching terms
We follow the notation of Matis et al:
a_n and b_n are linear and non-linear (feedback) terms, correspondingly.
x_1 and x_2 are birth and death parameters, respectively, for any x.
i is the immigration term

m is the multiplicity probability vector.


