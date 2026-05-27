import multiprocessing

from qiskit import QuantumCircuit
import torch
from qiskit.circuit.library import real_amplitudes
from qiskit.circuit import Parameter, QuantumCircuit, ParameterVector
import torch.nn as nn
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_machine_learning.gradients import SPSAEstimatorGradient
import numpy as np




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
    def __init__(self, n_qubits, n_dim, obs=None, gradient_mode='SPSA'):
        super(VQC, self).__init__()
        self.n_qubits = n_qubits
        self.quantum_ansats, self.d_padded, self.input_params, self.weight_params = quantumAnsatz(n_qubits, n_dim)

        #estimetor = StatevectorEstimator()

        estimator = AerEstimator()
        estimator.options.simulator = {
            "method": "statevector",  
            "device": "GPU",          # Usa la scheda video
            "cuStateVec_enable": True # Attiva i driver quantistici NVIDIA ad alte prestazioni
        }
        
        
        #Stessi risultati tra i due pesi iniziali, nessuno dei due mostra una convergenza più rapida. 
        #self.q_weights = nn.Parameter(torch.empty(len(list(self.weight_params))).uniform_(-0.01, 0.01))
        self.q_weights = nn.Parameter(torch.empty(len(list(self.weight_params))).uniform_(-np.pi, np.pi))

        gradient = ParamShiftEstimatorGradient(estimator)
        
        
        if gradient_mode == 'SPSA':
            self.gradient = SPSAEstimatorGradient(estimator)
        elif gradient_mode == 'SPSA_second_order':
            self.gradient = SPSAEstimatorGradient(estimator, second_order=True)
        else:
            raise ValueError(f"Modalità di gradiente non supportata: {gradient_mode}")
        
        #sampler = Sampler();
        #qnspsa = QNSPSA(maxiter=300, perturbation=0.01, learning_rate=0.01)

        # Create the Estimator QNN
        self.qnn = EstimatorQNN(circuit=self.quantum_ansats,
                                observables=obs,
                                input_params=self.input_params,
                                weight_params=self.weight_params,
                                estimator=estimator,
                                gradient=gradient,
                                input_gradients=False
                                )

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
    
    
    

    