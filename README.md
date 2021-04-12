# Quantum Computation

I will be implementing a variety of quantum algorithms in this repository. This repository uses Cirq and Tensorflow Quantum. I will be making videos on each of these when I find the time, and when I do the link will be here:

# Implemented Algorithms

## TensorFlow-Quantum (TFQ) and Cirq

Code for different TFQ experimentation. Includes original code and tutorials (and translated tutorials from pennylane to tfq). Video discussion on: https://www.youtube.com/channel/UC0U0HDNbdh0aI-9FbpYhPgg

Currently includes:

- Single Qubit Classifier

- Solving XOR with QML

- Replicating "Reinforcement learning with quantum variational circuits"

- Quantum Approximate Optimization Algorithm (QAOA) in TFQ

- Variational Quantum Eigensolver (VQE) in TFQ: include 1 and 2 qubit hamiltonians and replication of [Scalable Quantum Simulation of Molecular Energies](https://arxiv.org/pdf/1512.06860.pdf)

- Rotosolve Optimizer for VQEs in TFQ: from [Structure optimization for parameterized quantum circuits](https://quantum-journal.org/papers/q-2021-01-28-391/pdf/)

- VQE for arbitrarily many qubits in Cirq

- Custom ParameterShift and Adam optimization comparison with TFQ

- Arbitrary Qubit VQE in TFQ

- [SSVQE](https://arxiv.org/abs/1810.09434) for excited states in TFQ

- QOSF Application Problems:

  - Swap Test in Cirq

  - Simple Quantum Error Correction in Cirq

  - Quantum Simulator from Scratch

  - Weighted MaxCut QAOA in Cirq

- Barren Plateaus in TFQ

- Variational Quantum Classifiers/Regressors in TFQ for Circles, Moons, Blobs and Boston Housing

## Pennylane

Code for Pennylane experiments (largely from the [QHack](https://qhack.ai/) hackathon). Problems here: https://challenge.qhack.ai/team/problems. 

- Simple Circuits (20, 30, 50)

- Quantum Gradients (100, 200, 500)

- Circuit Training (100, 200, 500)

- Variational Quantum Eigensolvers (100, 200, 500)

## Research

Misc. code, may or may not run, definitely won't work. Just ideas and scraps that I figured I'd save.

## Quantum Teleportation

The quantum teleportation algorithm is a technique to transport a quantum state. If Alice and Bob have each have 1 qubit and they are entangled (in this case we consider them to be in a Bell state) then Alice can interact with her qubit and a message qubit and once she measures her qubit and sends some classical information to Bob, Bob knows the state of his qubit. Because common quantum simulations programs, including cirq, do not allow for operations after measurement the teleportation aspect is kind of lost. Nonetheless, the principle remains. 

## Deutsch–Jozsa Algorithm

In this folder I have both a classical (although it's faster because the function is discoverable very early because all 0s is 0 then add 1 and it returns 1, thus after 2 iterations it is discovered to be balanced, however this could be fixed with a better balanced function) and quantum algorithms. Basically, the toy problem is that you have {0,1}^n -> balanced or constant (i.e. even number of 1s and 0s or all 1s/0s). This requires 2^(n-1)+1 classical (worst case), but can be done much faster because the quantum superpositions can represent this much easier. I only have it implemented for n = 1 and n = 2. Adding more balanced would be easy, and constant likely would as well, but as this is such a toy example, I leave it at 2. 

## Grover's Algorithm

One of the most famous algorithms in quantum computing is also a relatively simple one. For a function, f(x) = 1 if x == ? else 0, this algorithm can provide O(sqrt(N)) time in searching. The way this algorithm works is by encoding everything, applying Hadamard gates, applying to oracle function, then applying the Grover operator. This grover operator basically flips everything around the average coefficient/measurement probability. Because the oracle function inverts the amplitude for the f(x) = 1 case, it is now negative. Thus flipping about the average decreases all but the specified one which is dramatically increased. However, this is probabilistic and needs to be done O(sqrt(N)) times. We simulate it many more times because of certain probabilistic unreliabilities. [Here](https://www.diva-portal.org/smash/get/diva2:1214481/FULLTEXT01.pdf) is a similar and beginner friendly overview. Note that, when using certain oracle, quantum uncomputation is needed (i.e. you need to undo the unitary operations done so there is no entangled interference). 

## Simon's Algorithm

This is another oracle based algorithm, and is similar to Deutsch-Jozsa in that it isn't the most practically applicable, but it demonstrates the power of quantum computing very well. Basically the problem is as such, there is a function, f(x), that meets the criteria f(x) = f(x XOR a) (for some binary string of length 2^n) and we need to find a. Classically we can query the oracle repeatedly until we have a match (i.e. f(x) = f(y)). From there we can do x XOR y and find a. However, worst case you have to query the oracle half of all possible inputs, which scales O(2^(n/2)) with n. This is not very good. The quantum way offers a significant (polynomial) speedup. We have 2n qubits in total. We apply Hadamard gates to the first n qubits (which we can think of as the algorithm qubits) then we generate the orcale via quantum gates with the next n qubits (the function qubits). These gates will vary depending on the oracle. In this case the oracle is simply all 1s (i.e. a = 111...1). Hadamard gates are reapplied to the first n qubits and measurements are taken of the first n. From there we have our answer. Note that this does need to be repeated (potentially) as all 0s is a trivial answer that will always be true (since x XOR 0 = x, f(x) = f(x XOR 0) for any function or input). We need to get the output besides all 0s so we may need to run it again. 


## Variational Quantum RL

These algorithms use variational quantum circuits to do reinforcement learning. The first examples comes from [Chen, et al. (2019)](https://arxiv.org/pdf/1907.00397.pdf), specifically frozen lake. These are programmed in the new (as of time of writing) Tensorflow Quantum. These can be thought of as similar to machine learning in that there are 'weights' that are being updated, specifically the parameters of the quantum circuits.
