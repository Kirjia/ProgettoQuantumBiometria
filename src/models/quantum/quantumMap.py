from qiskit import QuantumCircuit
import torch
import torch.nn.functional as F
from qiskit.circuit.library import real_amplitudes
from qiskit.circuit import QuantumCircuit, ParameterVector
import torch.nn as nn
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.primitives import BaseEstimatorV2 as BaseEstimator
from django.db import models

def zero_padding(n_qubits, n_dim):
    resto = n_dim % n_qubits
    if resto == 0:
        pad_size = 0
    else:
        pad_size = n_qubits - resto  # Nel caso 16 % 6 -> resto 4 -> servono 2 zeri

    d_padded = n_dim + pad_size          

    print(f"Dimensione originale: {n_dim}, Zeri aggiunti: {pad_size}, Dimensione padded: {d_padded}")
    return d_padded



class QuantumAnsats():
    def __init__(self, n_qubits, n_dim):
        self.n_qubits = n_qubits
        self.n_dim = n_dim
        self.d_padded = None
        self.x = None
        self.create_quantum_circuit(n_qubits, n_dim)

    def create_quantum_circuit(self,n_qubits, n_dim) -> tuple[QuantumCircuit, int]:
        self.d_padded = zero_padding(n_qubits, n_dim)
        ansats = real_amplitudes(num_qubits=6, reps=2, name="Ansats", entanglement="circular")
        ansats.draw("mpl")

        fm = QuantumCircuit(n_qubits)
        self.x = ParameterVector("x", length=self.d_padded)
        for i in range(self.d_padded):
            fm.ry(self.x[i], i % n_qubits)  # Applica la rotazione Ry al qubit i%n_qubits

        self.circ = fm.compose(ansats)
    
    @property
    def parameters(self):
        return self.x
    
    @property
    def circuit(self):
        return self.circ
    
    @property
    def padded_dimension(self):
        return self.d_padded
    
    @property
    def full_parameters(self):
        return self.circ.parameters



class VQC(nn.Module):
    def __init__(self, n_qubits, n_dim, estimator :BaseEstimator):
        super(VQC, self).__init__()
        self.quantum_ansats = QuantumAnsats(n_qubits, n_dim)
        initial_weights = torch.empty(estimator.num_weights).uniform_(-0.01, 0.01)
        self.quantum_layer = TorchConnector(estimator, initial_weights=initial_weights)
        self.linear = nn.Linear(n_qubits, 4)  # 4 classi di output
        
    def forward(self, x):
        quantum_output = self.quantum_layer(x)
        output = self.linear(quantum_output)
        return output
    
    