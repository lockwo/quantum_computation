import cirq
import random

'''
The bit-flip code and the sign-flip code (you can find a description of both here) are two very simple circuits able to detect and fix the bit-flip and the sign-flip errors, respectively.

Build the following simple circuit to prepare the Bell state: 

Now add, right before the CNOT gate and for each of the two qubits, an arbitrary “error gate”. 
By error gate we mean that with a certain probability (that you can decide but must be non-zero for all the choices) you have a 1 qubit unitary which can be either the identity, 
or the X gate (bit-flip error) or the Z gate (sign-flip error).

Encode each of the two qubits with a sign-flip or a bit-flip code, in such a way that all the possible choices for the error gates described in 2), 
occurring on the logical qubits, can be detected and fixed. Motivate your choice. This is the most non-trivial part of the problem, so do it with a lot of care!

Test your solution by making many measurements over the final state and testing that the results are in line with the expectations.
'''

qubits = [cirq.GridQubit(0, i) for i in range(2)]

sim = cirq.Simulator()

def bitstring(bits):
    return ''.join(str(int(b)) for b in bits)

bell_state = cirq.Circuit()
bell_state.append(cirq.H(qubits[0]))
bell_state.append(cirq.CNOT(qubits[0], qubits[1]))
bell_state.append(cirq.measure(*qubits, key='result'))
result = sim.run(bell_state, repetitions=100)
frequencies = result.histogram(key='result', fold_func=bitstring)
print(frequencies)

def error_bell(qubits):
    eb = cirq.Circuit()
    eb.append(cirq.H(qubits[0]))
    for i in range(2):
        r = random.random()
        if r < 1/3:
            eb.append(cirq.I(qubits[i]))
        elif r < 2/3:
            eb.append(cirq.X(qubits[i]))
        else:
            eb.append(cirq.Z(qubits[i]))
    eb.append(cirq.CNOT(qubits[0], qubits[1]))
    eb.append(cirq.measure(*qubits, key='result'))
    return eb

err = 0
n = 100
rep = 10
for i in range(n):
    cir = error_bell(qubits)
    result = sim.run(cir, repetitions=rep)
    frequencies = result.histogram(key='result', fold_func=bitstring)
    err = err + frequencies['01'] + frequencies['10']

print("Errors {}, Error Rate {}".format(err, err/(n * rep)))

def bit_flip_bell(qubits):
    eb = cirq.Circuit()
    bitflip_1 = [cirq.GridQubit(1, i) for i in range(2)]
    bitflip_2 = [cirq.GridQubit(2, i) for i in range(2)]
    eb.append(cirq.CNOT(qubits[0], bitflip_1[0]))
    eb.append(cirq.CNOT(qubits[0], bitflip_1[1]))
    eb.append(cirq.CNOT(qubits[1], bitflip_2[0]))
    eb.append(cirq.CNOT(qubits[1], bitflip_2[1]))
    eb.append(cirq.H(qubits[0]))
    for i in range(2):
        r = random.random()
        if r < 1/3:
            eb.append(cirq.I(qubits[i]))
        elif r < 2/3:
            eb.append(cirq.X(qubits[i]))
        else:
            eb.append(cirq.Z(qubits[i]))

    eb.append(cirq.CNOT(qubits[0], qubits[1]))

    eb.append(cirq.CNOT(qubits[0], bitflip_1[0]))
    eb.append(cirq.CNOT(qubits[0], bitflip_1[1]))
    eb.append(cirq.X(qubits[0]).controlled_by(*bitflip_1))
    eb.append(cirq.CNOT(qubits[1], bitflip_2[0]))
    eb.append(cirq.CNOT(qubits[1], bitflip_2[1]))
    eb.append(cirq.X(qubits[1]).controlled_by(*bitflip_2))
    eb.append(cirq.measure(*qubits, key='result'))
    return eb

tests = 1000
total_e = 0
errors_e = 0
total_b = 0
errors_b = 0

print(error_bell(qubits))
print(bit_flip_bell(qubits))

for i in range(tests):
    cir = error_bell(qubits)
    m = 10
    result = sim.run(cir, repetitions=m)
    frequencies = result.histogram(key='result', fold_func=bitstring)
    total_e += m
    errors_e = errors_e + frequencies['01'] + frequencies['10']
    bit = bit_flip_bell(qubits)
    result = sim.run(bit, repetitions=m)
    frequencies = result.histogram(key='result', fold_func=bitstring)
    total_b += m
    errors_b = errors_b + frequencies['01'] + frequencies['10']

print("Error Rate:", errors_e/total_e)
print("Bit flip error rate:", errors_b/total_b)
