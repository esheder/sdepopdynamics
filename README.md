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
We recommend working in a virtual environment, either through `virtualenv`, `conda`, or any other means.
To install, simply use `pip` with the command `pip install .` in the main directory of this repository 
or `pip install -e .` for an editable installation.

## How to use this code
We mostly expect researchers to want one of two things: Either they want to generate 
paths of the population distribution, or they want to sample from the asymptotic distribution.
We provide tools for both cases in both models.

Examples are given in the examples folder in both a Jupyter notebook and a Python script format.
We recommend looking over there first.

## Project Structure
Under the `popdynamics` directory you will find our package. 
For most applications, researchers will just need this directory and the `setup.py` file to get things started.

Under the `examples` directory you will find examples for how to use the code. These appear as both Python scripts and Jupyter notebooks.

Under the `tests` directory you will find our unit tests.
We could have more of these, and if we get more researchers interested in this code this would be higher own our to-do list.
You can consider contributing tests yourself, actually. It's a good place for new contributers to start.

## Implementation
Our implementation is currently done in Python. This could change later if found to be necessary.

