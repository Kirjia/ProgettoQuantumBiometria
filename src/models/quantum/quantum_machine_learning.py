import multiprocessing

from qiskit import QuantumCircuit
import torch
from qiskit.circuit import QuantumCircuit, ParameterVector
import torch.nn as nn
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import SPSAEstimatorGradient
from qiskit.primitives import StatevectorEstimator
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

import numpy as np
from typing import Any, Callable, Callable
from qiskit.quantum_info import SparsePauliOp

AnyFunction = Callable[..., Any]


def zero_padding(n_qubits, n_dim) -> tuple[int, int, int]:
    resto = n_dim % n_qubits
    if resto == 0:
        pad_size = 0
    else:
        pad_size = n_qubits - resto  # Nel caso 16 % 6 -> resto 4 -> servono 2 zeri

    d_padded = n_dim + pad_size  
    num_blocks = int(np.ceil(d_padded / n_qubits))


    return d_padded, num_blocks, pad_size
    
def pauli_observable(n_qubits, obs_str)-> list[str]:
    pauli = {'I': 'I', 'X': 'X', 'Y': 'Y', 'Z': 'Z'}
    observables = []
    string_identities = "I" * n_qubits
    for char in obs_str:
        for i in range(n_qubits):
            # Crea una stringa con l'osservabile specificato al posto del qubit i, e identità altrove
            obs = SparsePauliOp.from_sparse_list([(char, [i], 1.0)], num_qubits=n_qubits)
            observables.append(obs)
    return observables

#obsolete, non più usata, è stata sostituita da build_quantum_circuit che ora supporta anche ansats personalizzati e più livelli di encoding
'''def build_quantum_circuit(n_qubits, n_dim) -> tuple[QuantumCircuit, int, list, list]:
    d_padded = zero_padding(n_qubits, n_dim)
    ansats = real_amplitudes(num_qubits=n_qubits, reps=2, name="Ansatz")

    fm = QuantumCircuit(n_qubits)
    input_params = ParameterVector("x", length=d_padded)
    for i in range(d_padded):
        fm.ry(input_params[i], i % n_qubits)  # Applica la rotazione Ry al qubit i%n_qubits

    ansatz = fm.compose(ansats)
    weight_params = [param for param in ansatz.parameters if param.name.startswith("θ")]
    
    return ansatz, d_padded, input_params, weight_params'''
       

from qiskit import transpile

class VQC(nn.Module):




    def __init__(self, n_qubits: int, quantum_circuit: QuantumCircuit, obs: list[str], gradient_mode='SPSA', input_params=None, weight_params=None, target_classes = 2, **kwargs):
        super(VQC, self).__init__()
        self.n_qubits = n_qubits
        self.quantum_circuit = transpile(quantum_circuit,
                                            optimization_level=3, 
                                            basis_gates=['u', 'cx', 'ry', 'rz'])
        self.input_params = input_params
        self.weight_params = weight_params


        #estimator = StatevectorEstimator()

        estimator = AerEstimator()
        estimator.options.simulator = {
            "method": "statevector",  
            "device": "GPU",          # Usa la scheda video
            "cuStateVec_enable": True # Attiva i driver quantistici NVIDIA ad alte prestazioni
        }
        
        
        #Stessi risultati tra i due pesi iniziali, nessuno dei due mostra una convergenza più rapida. 
        self.q_weights = nn.Parameter(torch.empty(len(list(self.weight_params))).uniform_(-0.01, 0.01))
        #self.q_weights = nn.Parameter(torch.empty(len(list(self.weight_params))).uniform_(-np.pi, np.pi))

        
        
        if gradient_mode == 'SPSA':
            self.gradient = SPSAEstimatorGradient(estimator, epsilon=0.01)
        elif gradient_mode == 'SPSA_second_order':
            self.gradient = SPSAEstimatorGradient(estimator, epsilon=0.01, second_order=True)
        else:
            raise ValueError(f"Modalità di gradiente non supportata: {gradient_mode}")
        
      

        # Create the Estimator QNN
        self.qnn = EstimatorQNN(circuit=self.quantum_circuit,
                                observables=obs,
                                input_params=self.input_params,
                                weight_params=self.weight_params,
                                estimator=estimator,
                                gradient=self.gradient,
                                input_gradients=False
                                )

        # Connect to PyTorch
        self.quantum_layer = TorchConnector(self.qnn, initial_weights=self.q_weights)

        # Classical output layer
        num_observables = len(obs)
        self.head_classical_linear_layer = nn.Linear(num_observables, target_classes)  # Output layer per la classificazione finale

    def forward(self, x):
        q_out = self.quantum_layer(x)

        
        # 5. Classificazione finale
        logits = self.head_classical_linear_layer(q_out)
        return logits
    


def build_ansatz(ansatz_fun: AnyFunction, n_qubits: int,
                 reps: int,
                 name: str ,
                 **kwargs: Any) -> QuantumCircuit:
    return ansatz_fun(num_qubits=n_qubits, reps=reps, name=name, **kwargs)



def build_quantum_circuit(real_qubits: int,
                          encoding_depth: int, **kwargs) -> dict[str, Any]:
    
    total_qubits = encoding_depth * real_qubits
    input_params = ParameterVector("x", length=total_qubits)
    quantum_circuit = QuantumCircuit(real_qubits)
    weight_params = []
    
    reps = kwargs.get('reps', 1)  # Ottieni il numero di ripetizioni, default a 1 se non specificato
    kwargs.pop('reps', None)

    for layer in range(encoding_depth):
        print(f"Costruzione del layer {layer} con {real_qubits} qubits reali e {total_qubits} qubits totali")
        for i in range(real_qubits):
            idx = layer * real_qubits + i
            quantum_circuit.ry(input_params[idx], i)  # Applica la rotazione Ry al qubit i%total_qubits
        
        
        ansatz = build_ansatz(n_qubits=real_qubits, reps=reps, name=f'ansatz_{layer}', parameter_prefix=f'θ_{layer}', **kwargs)
        quantum_circuit.compose(ansatz, inplace=True)
        weight_params.extend(ansatz.parameters)
    return {'quantum_circuit': quantum_circuit,
             'weight_params': weight_params,
             'input_params': input_params}

    
        