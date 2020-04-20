# Quantum Computation

I will be implementing a variety of quantum algorithms in this repository. I will start off with simple well established ones as I learn about the process, then I will experiment with some new ones. 

# Implemented Algorithms

## Deutsch–Jozsa algorithm

In this folder I have both a classical (although it's faster because the function is discoverable very early because all 0s is 0 then add 1 and it returns 1, thus after 2 iterations it is discovered to be balanced, however this could be fixed with a better balanced function) and quantum algorithms. Basically, the toy problem is that you have {0,1}^n -> balanced or constant (i.e. even number of 1s and 0s or all 1s/0s). This requires 2^(n-1)+1 classical (worst case), but can be done much faster because the quantum superpositions can represent this much easier. I only have it implemented for n = 1 and n = 2. Adding more balanced would be easy, and constant likely would as well, but as this is such a toy example, I leave it at 2. 

## Grover's Algorithm

One of the most famous algorithms in quantum computing is also a relatively simple one. For a function, f(x) = 1 if x == ? else 0, this algorithm can provide O(sqrt(N)) time in searching. The way this algorithm works is by encoding everything, applying Hadamard gates, applying to oracle function, then applying the Grover operator. This grover operator basically flips everything around the average coefficient/measurement probability. Because the oracle function inverts the amplitude for the f(x) = 1 case, it is now negative. Thus flipping about the average decreases all but the specified one which is dramatically increased. However, this is probabilistic and needs to be done O(sqrt(N)) times. We simulate it many more times because of certain probabilistic unreliabilities. [Here](https://www.diva-portal.org/smash/get/diva2:1214481/FULLTEXT01.pdf) is a similar and beginner friendly overview. 

## Variational Quantum RL

These algorithms use variational quantum circuits to do reinforcement learning. The first examples comes from [Chen, et al. (2019)](https://arxiv.org/pdf/1907.00397.pdf), specifically frozen lake. These are programmed in the new (as of time of writing) Tensorflow Quantum. These can be thought of as similar to machine learning in that there are 'weights' that are being updated, specifically the parameters of the quantum circuits.
