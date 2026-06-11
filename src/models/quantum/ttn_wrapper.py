from typing import Any
from qiskit.circuit import QuantumCircuit, ParameterVector

from .ttn_circuit import build_ttn_circuit


def ttn_ansatz(num_qubits: int, reps: int = 1, name: str = "ttn", parameter_prefix: str | None = None, **kwargs: Any) -> QuantumCircuit:
    """Adapts the project's TTN builder to the ansatz_fun signature expected by build_quantum_circuit.

    If `parameter_prefix` is provided, rebinds internal parameters to a new
    `ParameterVector` so external code can track them with a stable name.
    """
    qc = build_ttn_circuit(num_qubits, reps=reps, name=name)

    old_params = list(qc.parameters)
    if parameter_prefix is None or not old_params:
        return qc

    new_params = ParameterVector(parameter_prefix, length=len(old_params))
    mapping = {old: new for old, new in zip(old_params, new_params)}
    return qc.assign_parameters(mapping, inplace=False)
