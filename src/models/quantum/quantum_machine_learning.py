
from qiskit import QuantumCircuit
import torch
from qiskit.circuit import QuantumCircuit, ParameterVector
import torch.nn as nn

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import SPSAEstimatorGradient
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

import json
import yaml
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
from typing import Any, Callable, Callable
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import real_amplitudes
from qiskit.circuit.library import zz_feature_map
from qiskit.circuit.library import efficient_su2

AnyFunction = Callable[..., Any]


from dataclasses import dataclass
from utils import ASSETS_DIR

@dataclass
class ExperimentConfig:
    # --- Architettura Quantistica ---
    n_qubits: int = 6
    input_dim: int = 32
    reps: int = 2
    ansatz_function: str = "real_amplitudes"
    encoding_function: str = "ry"
    readout_name: str = "z"
    
    # --- Machine Learning ---
    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 1e-3
    num_classes: int = 4
    n_splits_kfold: int = 5
    early_stopping_patience: int = 5

    encoding_range: tuple[float, float] = (0, 2 * np.pi)  # Range per la normalizzazione dei dati di input
    
    # --- Hardware ---
    use_gpu: bool = True
    seed: int = 11


def salva_esperimento_locale(cfg, ansatz_kwargs, report, y_true_test=None, y_pred_test=None, base_dir=ASSETS_DIR / "experiments"):
    """
    Salva in modo professionale e locale le configurazioni e le metriche dell'esperimento Quantum.
    """
   
    run_name = f"VQC_q{cfg.n_qubits}_dim{cfg.input_dim}_seed_{cfg.seed}_ansatz{cfg.ansatz_function}_key_{hashlib.md5(str(cfg.__dict__).encode()).hexdigest()[:8]}"
    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = cfg.__dict__.copy() if hasattr(cfg, '__dict__') else dict(cfg)
    if "encoding_range" in cfg_dict and isinstance(cfg_dict["encoding_range"], tuple):
        cfg_dict["encoding_range"] = list(cfg_dict["encoding_range"])

    full_config = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": run_name,
        },
        "experiment_config": cfg_dict,
        "ansatz_kwargs": ansatz_kwargs,
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)

    final_metrics = {}
    if report.get("val_f1"):
        final_metrics["max_val_f1"] = float(np.max(report["val_f1"]))
        final_metrics["mean_val_f1"] = float(np.mean(report["val_f1"]))
        final_metrics["final_val_loss"] = float(report["val_loss"][-1])
        final_metrics["final_train_loss"] = float(report["train_loss"][-1])
        final_metrics["total_time_sec"] = float(np.sum(report["epoch_time"]))
    if report.get("val_auc"):
        final_metrics["max_val_auc"] = float(np.max(report["val_auc"]))
        final_metrics["mean_val_auc"] = float(np.mean(report["val_auc"]))
        final_metrics["final_val_auc"] = float(report["val_auc"][-1])
    if report.get("val_ece"):
        final_metrics["min_val_ece"] = float(np.min(report["val_ece"]))
        final_metrics["mean_val_ece"] = float(np.mean(report["val_ece"]))
        final_metrics["final_val_ece"] = float(report["val_ece"][-1])

    test_metrics = {}
    if y_true_test is not None and y_pred_test is not None:
        from sklearn.metrics import f1_score, classification_report
        test_metrics["macro_f1"] = float(f1_score(y_true_test, y_pred_test, average='macro'))
        test_metrics["classification_report"] = classification_report(y_true_test, y_pred_test, output_dict=True)
    if report.get("test_auc") is not None:
        test_metrics["auc"] = float(report["test_auc"])
    if report.get("test_ece") is not None:
        test_metrics["ece"] = float(report["test_ece"])

    num_epochs = len(report.get("val_loss", []))
    epoch_history = []
    epochs = report.get("epoch", list(range(1, num_epochs + 1)))
    val_class_reports = report.get("val_classification_reports", [None] * num_epochs)
    val_aucs = report.get("val_auc", [None] * num_epochs)
    val_eces = report.get("val_ece", [None] * num_epochs)
    
    for idx in range(num_epochs):
        epoch_history.append({
            "epoch": int(epochs[idx]) if idx < len(epochs) else idx + 1,
            "val_loss": float(report["val_loss"][idx]) if "val_loss" in report else None,
            "val_f1": float(report["val_f1"][idx]) if "val_f1" in report else None,
            "train_loss": float(report["train_loss"][idx]) if "train_loss" in report else None,
            "epoch_time_sec": float(report["epoch_time"][idx]) if "epoch_time" in report else None,
            "val_auc": float(val_aucs[idx]) if idx < len(val_aucs) and val_aucs[idx] is not None else None,
            "val_ece": float(val_eces[idx]) if idx < len(val_eces) and val_eces[idx] is not None else None,
            "val_classification_report": val_class_reports[idx] if idx < len(val_class_reports) else None,
        })

    output_metrics = {
        "metadata": {
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": run_name,
        },
        "summary": final_metrics,
        "history": epoch_history,
    }

    if test_metrics:
        output_metrics["test"] = test_metrics

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output_metrics, f, indent=4)

    print(f"✅ Esperimento salvato localmente con successo!")
    print(f"📂 Cartella: {run_dir}")

    return run_dir


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
       


'''###############################################################
#######################  PARTE REVISIONATA  ########################
###############################################################'''
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
        #######################  PARTE REVISIONATA  ########################
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
        #######################  PARTE REVISIONATA  ########################
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



"""
    Encloses any type of quantum circuit construction, allowing for flexible ansatz design and multiple encoding layers.
        - build_ansatz: A helper function to construct the ansatz based on a provided function and parameters.

        Parameters: [ansatz_fun: callable, n_qubits: int, reps: int, name: str, **kwargs: Any]
        >> ansatz_fun: A callable that generates a quantum circuit ansatz (e.g., real_amplitudes).
        >> n_qubits: The number of qubits to use in the ansatz.
        >> reps: The number of repetitions (layers) in the ansatz.
        >> name: A string to name the ansatz for parameter identification.
        >> **kwargs: Additional parameters that may be required by the ansatz function.

"""
def build_ansatz(ansatz_fun: AnyFunction, n_qubits: int,
                 reps: int,
                 name: str,
                 **kwargs: Any) -> QuantumCircuit:
    return ansatz_fun(num_qubits=n_qubits, reps=reps, name=name, **kwargs)


def default_encoding_layer(n_qubits: int, input_params: ParameterVector, layer: int = 0, **kwargs: Any) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    for i, param in enumerate(input_params):
        qc.ry(param, i)
    return qc


def rx_encoding_layer(n_qubits: int, input_params: ParameterVector, layer: int = 0, **kwargs: Any) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    for i, param in enumerate(input_params):
        if i % 2 == 0:
            qc.ry(param, i)
        else:
            qc.rz(param, i)
       
    return qc


def build_encoding_layer(encoding_fun: AnyFunction, n_qubits: int, input_params: ParameterVector, layer: int = 0, **kwargs: Any) -> QuantumCircuit:
    return encoding_fun(n_qubits=n_qubits, layer=layer, input_params=input_params, **kwargs)


def zzfeaturemap_encoding_layer(
    n_qubits: int,
    layer: int = 0,
    input_params: ParameterVector | None = None,
    parameter_prefix: str = "x",
    reps: int = 1,
    data_map_func: Callable[[np.ndarray, int], float] | None = None,
    **kwargs: Any,
) -> QuantumCircuit:
    circuit = zz_feature_map(
        feature_dimension=n_qubits,
        reps=reps,
        name=f'zzfeaturemap_layer{layer}',
        parameter_prefix=parameter_prefix,
        data_map_func=data_map_func,
        **kwargs,
    )
    if input_params is not None:
        fm_params = list(circuit.parameters)
        if len(fm_params) != len(input_params):
            raise ValueError(
                f"zz_feature_map created {len(fm_params)} parameters, but {len(input_params)} were provided."
            )
        circuit = circuit.assign_parameters(
            {fm_param: input_param for fm_param, input_param in zip(fm_params, input_params)}
        )
    return circuit


        
    

ENCODING_LAYER_FACTORY: dict[str, AnyFunction] = {
    'ry': default_encoding_layer,
    'rx': rx_encoding_layer,
    'zzfeaturemap': zzfeaturemap_encoding_layer,
}

ANSATZ_FACTORY: dict[str, AnyFunction] = {
    'real_amplitudes': real_amplitudes,
    'efficient_su2': efficient_su2,
    'ttn': None,  # placeholder: resolved at import time to avoid circular import
}

# Lazy import: attach TTN ansatz if available
try:
    from .ttn_wrapper import ttn_ansatz
    ANSATZ_FACTORY['ttn'] = ttn_ansatz
except Exception:
    # If import fails (missing qiskit or file), leave as None and let callers pass ansatz_fun directly
    pass


def build_quantum_circuit(
    real_qubits: int,
    encoding_depth: int,
    *,
    ansatz_name: str = 'real_amplitudes',
    encoding_name: str = 'ry',
    ansatz_fun: AnyFunction | None = None,
    encoding_fun: AnyFunction | None = None,
    reps: int = 1,
    ansatz_kwargs: dict[str, Any] | None = None,
    encoding_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    
    total_qubits = encoding_depth * real_qubits
    input_params = ParameterVector("x", length=total_qubits)
    quantum_circuit = QuantumCircuit(real_qubits)
    weight_params = []

    ansatz_kwargs = ansatz_kwargs or {}
    encoding_kwargs = encoding_kwargs or {}

    ansatz_kwargs.update(kwargs.pop('ansatz_kwargs', {}) or {})
    encoding_kwargs.update(kwargs.pop('encoding_kwargs', {}) or {})

    if encoding_fun is None:
        try:
            encoding_fun = ENCODING_LAYER_FACTORY[encoding_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown encoding_name {encoding_name!r}. Supported names: {list(ENCODING_LAYER_FACTORY)}"
            ) from exc

    if ansatz_fun is None:
        try:
            ansatz_fun = ANSATZ_FACTORY[ansatz_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown ansatz_name {ansatz_name!r}. Supported names: {list(ANSATZ_FACTORY)}"
            ) from exc

    if kwargs:
        # Allow legacy constructor kwargs for encoding and ansatz layers.
        encoding_kwargs.update(kwargs)
        

    for layer in range(encoding_depth):
        print(f"Costruzione del layer {layer} con {real_qubits} qubits reali e {total_qubits} qubits totali")
        start = layer * real_qubits
        end = start + real_qubits
        layer_encoding = build_encoding_layer(
            encoding_fun=encoding_fun,
            n_qubits=real_qubits,
            layer=layer,
            input_params=input_params[start:end],
            **encoding_kwargs,
        )
        quantum_circuit.compose(layer_encoding, inplace=True)

        ansatz = build_ansatz(
            ansatz_fun,
            real_qubits,
            reps,
            name=f'ansatz_{layer}',
            parameter_prefix=f"θ_{layer}",
            **ansatz_kwargs,
        )
        quantum_circuit.compose(ansatz, inplace=True)
        weight_params.extend(ansatz.parameters)

    return {
        'quantum_circuit': quantum_circuit,
        'weight_params': weight_params,
        'input_params': input_params,
    }

        
