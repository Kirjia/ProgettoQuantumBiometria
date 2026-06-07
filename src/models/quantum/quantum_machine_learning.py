
from qiskit import QuantumCircuit
import torch
from qiskit.circuit import QuantumCircuit, ParameterVector
import torch.nn as nn

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import SPSAEstimatorGradient
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

import numpy as np
from typing import Any, Callable, Callable
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import real_amplitudes

AnyFunction = Callable[..., Any]


from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    # --- Architettura Quantistica ---
    n_qubits: int = 6
    input_dim: int = 32
    reps: int = 2
    ansatz_function: AnyFunction = real_amplitudes
    readout_name: str = "z"
    
    # --- Machine Learning ---
    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 1e-3
    num_classes: int = 4
    n_splits_kfold: int = 5
    
    # --- Hardware ---
    use_gpu: bool = True
    seed: int = 11


# Metodo padding. 
# Input: numero qubits, dimensione dati in input. Output: resto suddivisione padding, 
# numero blocchi ottenuti, e grandezza padding
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



'''###############################################################
#######################  PARTE FATTA DAL TUTOR  ########################
###############################################################'''


# Invece della backprop standard, l'algoritmo SPSA (o QNSPSA) "scuote" i 
# parametri quantistici per stimare la direzione del gradiente
class _BatchedEstimatorPauliSPSAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, model: "VQC", x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        features = model._estimate_readout_features(
            x.detach().cpu().numpy().astype(np.float64),
            weights.detach().cpu().numpy().astype(np.float64),
        )
        ctx.model = model
        ctx.save_for_backward(x.detach(), weights.detach())
        return torch.tensor(features, dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weights = ctx.saved_tensors
        model: VQC = ctx.model
        grad_weights = model._spsa_weight_gradient(
            x.detach().cpu().numpy().astype(np.float64),
            weights.detach().cpu().numpy().astype(np.float64),
            grad_output.detach().cpu().numpy().astype(np.float64),
        )
        return None, None, torch.tensor(grad_weights, dtype=weights.dtype, device=weights.device)
'''###############################################################'''

class VQC(nn.Module):




    def __init__(self, n_qubits: int, quantum_circuit: QuantumCircuit, obs: list[str], gradient_mode='SPSA', input_params=None, weight_params=None, target_classes = 2, **kwargs):
        super(VQC, self).__init__()
        self.n_qubits = n_qubits
        self.quantum_circuit = quantum_circuit
        
        self.input_params = input_params
        self.weight_params = weight_params


        #estimator = StatevectorEstimator()

        estimator = AerEstimator()
        simulator_options = {"method": "statevector"}
        if kwargs.get('use_gpu', True):
            simulator_options.update({
                "device": "GPU",          # Usa la scheda video
                "cuStateVec_enable": True # Attiva i driver quantistici NVIDIA ad alte prestazioni
            })
        estimator.options.simulator = simulator_options
        
        
        #Stessi risultati tra i due pesi iniziali, nessuno dei due mostra una convergenza più rapida. 
        self.q_weights = nn.Parameter(torch.empty(len(list(self.weight_params))).uniform_(-0.01, 0.01))
        #self.q_weights = nn.Parameter(torch.empty(len(list(self.weight_params))).uniform_(-np.pi, np.pi))

        
        num_observables = len(obs)
        self.head_classical_linear_layer = nn.Linear(num_observables, target_classes)  # Output layer per la classificazione finale

        '''###############################################################
        #######################  PARTE FATTA DAL TUTOR  ########################
        ###############################################################'''
        self._use_batched_estimator_spsa = gradient_mode == 'estimator_pauli_batched_spsa'

        if self._use_batched_estimator_spsa:
            self.estimator = estimator
            self.observables = [[observable] for observable in obs]
            self.readout_dim = num_observables
            self.estimator_precision = float(kwargs.get('estimator_precision', 0.0))
            self.spsa_epsilon = float(kwargs.get('spsa_epsilon', 1e-6))
            self.spsa_batch_size = int(kwargs.get('spsa_batch_size', 1))
            if self.spsa_epsilon <= 0:
                raise ValueError(f"spsa_epsilon deve essere > 0, trovato: {self.spsa_epsilon}")
            if self.spsa_batch_size <= 0:
                raise ValueError(f"spsa_batch_size deve essere > 0, trovato: {self.spsa_batch_size}")
            self._spsa_rng = np.random.default_rng(int(kwargs.get('seed', 0)))

            source_params = list(self.input_params) + list(self.weight_params)
            source_index = {param: index for index, param in enumerate(source_params)}
            self._parameter_order = list(self.quantum_circuit.parameters)
            try:
                self._parameter_source_indices = np.array(
                    [source_index[param] for param in self._parameter_order],
                    dtype=np.int64,
                )
            except KeyError as exc:
                raise RuntimeError("L'ordine dei parametri del circuito contiene parametri non tracciati.") from exc
            return
        '''###############################################################'''
        
        if gradient_mode == 'SPSA':
            self.gradient = SPSAEstimatorGradient(estimator, epsilon=0.01)
        elif gradient_mode == 'SPSA_second_order':
            self.gradient = SPSAEstimatorGradient(estimator, epsilon=0.01, second_order=True)
        elif gradient_mode == 'estimator_pauli_batched_spsa': #'''QUESTA VA INSERITA ALTRIMENTI NELLA FUNZIONE FORWARD, NON PUÒ ESSERE USATA COME GRADIENTE DIRETTO '''
            self.gradient = None  # Gestito manualmente nella funzione autograd personalizzata
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

        
   
    def forward(self, x):    # x è un tensore di forma (batch_size, n_dim) con i dati di input
        if self._use_batched_estimator_spsa:
            '''Utilizza la funzione autograd personalizzata per il gradiente SPSA con EstimatorV2 fatta dal tutor'''
            q_out = _BatchedEstimatorPauliSPSAFunction.apply(self, x, self.q_weights) 
        else:
            q_out = self.quantum_layer(x)

        
        # 5. Classificazione finale
        logits = self.head_classical_linear_layer(q_out)
        return logits
    
    '''###############################################################
        #######################  PARTE FATTA DAL TUTOR  ########################
        ###############################################################'''
    def _ordered_parameter_values(self, input_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if input_values.ndim == 1:
            input_values = input_values.reshape(1, -1)
        if weights.ndim == 1:
            weight_values = np.broadcast_to(weights, (input_values.shape[0], weights.shape[0]))
        else:
            weight_values = weights
        source_values = np.concatenate([input_values, weight_values], axis=1)
        return source_values[:, self._parameter_source_indices]

    def _estimate_readout_features(self, input_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        parameter_values = self._ordered_parameter_values(input_values, weights)
        pub = (self.quantum_circuit, self.observables, parameter_values)
        result = self.estimator.run([pub], precision=self.estimator_precision).result()
        evs = np.asarray(result[0].data.evs, dtype=np.float64)
        expected_size = self.readout_dim * input_values.shape[0]
        if evs.size != expected_size:
            raise RuntimeError(
                f"EstimatorV2 ha restituito evs size={evs.size}; atteso {expected_size} "
                f"per readout_dim={self.readout_dim} e batch={input_values.shape[0]}."
            )
        return evs.reshape(self.readout_dim, input_values.shape[0]).T

    def _spsa_weight_gradient(
        self,
        input_values: np.ndarray,
        weights: np.ndarray,
        upstream_gradient: np.ndarray,
    ) -> np.ndarray:
        deltas = self._spsa_rng.choice(
            np.array([-1.0, 1.0], dtype=np.float64),
            size=(self.spsa_batch_size, weights.shape[0]),
        )
        perturbed = []
        for delta in deltas:
            perturbed.append(weights + self.spsa_epsilon * delta)
            perturbed.append(weights - self.spsa_epsilon * delta)
        perturbed_weights = np.repeat(np.stack(perturbed, axis=0), input_values.shape[0], axis=0)
        repeated_inputs = np.tile(input_values, (len(perturbed), 1))
        features = self._estimate_readout_features(repeated_inputs, perturbed_weights)
        features = features.reshape(len(perturbed), input_values.shape[0], -1)

        grad = np.zeros_like(weights, dtype=np.float64)
        for index, delta in enumerate(deltas):
            plus = features[2 * index]
            minus = features[2 * index + 1]
            directional = np.sum(upstream_gradient * (plus - minus)) / (2.0 * self.spsa_epsilon)
            grad += directional * delta
        return grad / float(self.spsa_batch_size)
    '''###############################################################'''


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