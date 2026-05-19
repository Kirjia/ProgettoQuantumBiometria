import multiprocessing

from qiskit import QuantumCircuit
import torch
from qiskit.circuit.library import real_amplitudes
from qiskit.circuit import Parameter, QuantumCircuit, ParameterVector
import torch.nn as nn
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as EstimatorAer
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

import os

os.environ['OMP_NUM_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '12'
os.environ['OPENBLAS_NUM_THREADS'] = '12'



def zero_padding(n_qubits, n_dim):
    resto = n_dim % n_qubits
    if resto == 0:
        pad_size = 0
    else:
        pad_size = n_qubits - resto  # Nel caso 16 % 6 -> resto 4 -> servono 2 zeri

    d_padded = n_dim + pad_size          

    print(f"Dimensione originale: {n_dim}, Zeri aggiunti: {pad_size}, Dimensione padded: {d_padded}")
    return d_padded



def quantumAnsatz(n_qubits, n_dim) -> tuple[QuantumCircuit, int, list, list]:
    d_padded = zero_padding(n_qubits, n_dim)
    ansats = real_amplitudes(num_qubits=6, reps=2, name="Ansats")

    fm = QuantumCircuit(n_qubits)
    input_params = ParameterVector("x", length=d_padded)
    for i in range(d_padded):
        fm.ry(input_params[i], i % n_qubits)  # Applica la rotazione Ry al qubit i%n_qubits

    ansatz = fm.compose(ansats)
    weight_params = [param for param in ansatz.parameters if param.name.startswith("θ")]
    
    
    return ansatz, d_padded, input_params, weight_params
       



class VQC(nn.Module):
    def __init__(self, n_qubits, n_dim, obs=None):
        super(VQC, self).__init__()
        self.n_qubits = n_qubits
        self.quantum_ansats, self.d_padded, self.input_params, self.weight_params = quantumAnsatz(n_qubits, n_dim)

        #estimetor = StatevectorEstimator()

        estimator = AerEstimator()
        estimator.options.simulator = {
            "method": "statevector",  # Simulazione esatta richiesta dal prof
            "device": "GPU",          # Usa la scheda video
            "cuStateVec_enable": True # Attiva i driver quantistici NVIDIA ad alte prestazioni
        }
        
        

        self.q_weights = nn.Parameter(torch.empty(len(list(self.quantum_ansats.parameters))).uniform_(-0.01, 0.01))
        
        gradients = ParamShiftEstimatorGradient(estimator)

        # Create the Estimator QNN
        self.qnn = EstimatorQNN(circuit=self.quantum_ansats,
                                observables=obs,
                                input_params=self.input_params,
                                weight_params=self.weight_params,
                                estimator=estimator,
                                gradient=gradients,
                                input_gradients=True)

        # Connect to PyTorch
        self.quantum_layer = TorchConnector(self.qnn)

        # Classical output layer
        num_observables = len(obs) if obs else n_qubits
        self.linear = nn.Linear(num_observables, 4)

    def forward(self, x):
        q_out = self.quantum_layer(x)

        
        # 5. Classificazione finale
        logits = self.linear(q_out)
        return logits
    
    