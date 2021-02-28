#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def find_excited_states(H):
    """
    Fill in the missing parts between the # QHACK # markers below. Implement
    a variational method that can find the three lowest energies of the provided
    Hamiltonian.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The lowest three eigenenergies of the Hamiltonian as a comma-separated string,
        sorted from smallest to largest.
    """

    energies = np.zeros(3)

    # QHACK #
    np.random.seed(52)

    def variational_ansatz(params, wires):
        for k in range(len(params)):
            for i in range(len(params[k])):
                qml.Rot(params[k][i][0], params[k][i][1], params[k][i][2], wires=i)

            for i in range(len(params[k]) - 1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[len(wires) - 1, 0])

    def variational_ansatz1(params, wires):
        qml.PauliX(wires=1)
        for k in range(len(params)):
            for i in range(len(params[k])):
                qml.Rot(params[k][i][0], params[k][i][1], params[k][i][2], wires=i)

            for i in range(len(params[k]) - 1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[len(wires) - 1, 0])

    def variational_ansatz2(params, wires):
        qml.PauliX(wires=0)
        for k in range(len(params)):
            for i in range(len(params[k])):
                qml.Rot(params[k][i][0], params[k][i][1], params[k][i][2], wires=i)

            for i in range(len(params[k]) - 1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[len(wires) - 1, 0])
    
    num_qubits = len(H.wires)
   
    energy = 0
    opt = qml.AdagradOptimizer(0.2)

    max_iterations = 500
    conv_tol = 1e-04

    dev = qml.device('default.qubit', wires=num_qubits)
    dev1 = qml.device('default.qubit', wires=num_qubits)
    dev2 = qml.device('default.qubit', wires=num_qubits)
    cost_fn = qml.ExpvalCost(variational_ansatz, H, dev)
    cost_fn1 = qml.ExpvalCost(variational_ansatz1, H, dev1)
    cost_fn2 = qml.ExpvalCost(variational_ansatz2, H, dev2)

    def cost(params):
        return 0.55 * cost_fn2(params) + 0.65 * cost_fn1(params) + cost_fn(params)

    prev_energy = 100
    params = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(6, num_qubits, 3))
    for n in range(max_iterations):
        params, c = opt.step_and_cost(cost, params)
        energy = cost(params)
        conv = np.abs(energy - prev_energy)
        prev_energy = energy

        #if n % 20 == 0:
        #    print(energy)
        #    print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, cost_fn(params)))
        #   print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, cost_fn1(params)))
        #    print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, cost_fn2(params)))
            
        if conv <= conv_tol:
            break

    energies[0] = cost_fn(params)
    energies[1] = cost_fn1(params)
    energies[2] = cost_fn2(params)

    # QHACK #

    return ",".join([str(E) for E in energies])


def pauli_token_to_operator(token):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Helper function to turn strings into qml operators.

    Args:
        token (str): A Pauli operator input in string form.

    Returns:
        A qml.Operator instance of the Pauli.
    """
    qubit_terms = []

    for term in token:
        # Special case of identity
        if term == "I":
            qubit_terms.append(qml.Identity(0))
        else:
            pauli, qubit_idx = term[0], term[1:]
            if pauli == "X":
                qubit_terms.append(qml.PauliX(int(qubit_idx)))
            elif pauli == "Y":
                qubit_terms.append(qml.PauliY(int(qubit_idx)))
            elif pauli == "Z":
                qubit_terms.append(qml.PauliZ(int(qubit_idx)))
            else:
                print("Invalid input.")

    full_term = qubit_terms[0]
    for term in qubit_terms[1:]:
        full_term = full_term @ term

    return full_term


def parse_hamiltonian_input(input_data):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Turns the contents of the input file into a Hamiltonian.

    Args:
        filename(str): Name of the input file that contains the Hamiltonian.

    Returns:
        qml.Hamiltonian object of the Hamiltonian specified in the file.
    """
    # Get the input
    coeffs = []
    pauli_terms = []

    # Go through line by line and build up the Hamiltonian
    for line in input_data.split("S"):
        line = line.strip()
        tokens = line.split(" ")

        # Parse coefficients
        sign, value = tokens[0], tokens[1]

        coeff = float(value)
        if sign == "-":
            coeff *= -1
        coeffs.append(coeff)

        # Parse Pauli component
        pauli = tokens[2:]
        pauli_terms.append(pauli_token_to_operator(pauli))

    return qml.Hamiltonian(coeffs, pauli_terms)


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Turn input to Hamiltonian
    H = parse_hamiltonian_input(sys.stdin.read())

    # Send Hamiltonian through VQE routine and output the solution
    lowest_three_energies = find_excited_states(H)
    print(lowest_three_energies)
