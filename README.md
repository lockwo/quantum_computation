# Quantum Computation

I will be implementing a variety of quantum algorithms in this repository. I will start off with simple well established ones as I learn about the process, then I will experiment with some new ones. 

# Implemented Algorithms

## Deutsch–Jozsa algorithm

In this folder I have both a classical (although it's faster because the function is discoverable very early because all 0s is 0 then add 1 and it returns 1, thus after 2 iterations it is discovered to be balanced, however this could be fixed with a better balanced function) and quantum algorithms. Basically, the toy problem is that you have {0,1}^n -> balanced or constant (i.e. even number of 1s and 0s or all 1s/0s). This requires 2^(n-1)+1 classical (worst case), but can be done much faster because the quantum superpositions can represent this much easier. I only have it implemented for n = 1 and n = 2. Adding more balanced would be easy, and constant likely would as well, but as this is such a toy example, I leave it at 2. 
