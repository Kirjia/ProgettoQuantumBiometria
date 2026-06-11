#!/usr/bin/env python3
"""
Builder per circuiti Tree Tensor Network (TTN) in Qiskit, scalabile a n qubit.
Esempio d'uso:
    from src.models.quantum.ttn_circuit import build_ttn_circuit
    qc = build_ttn_circuit(8, reps=3)
    print(qc.draw(output='text'))

Questa implementazione costruisce un TTN binario: applica rotazioni locali, un
blocco entangliante a coppie di qubit e poi conserva un qubit per coppia per il
livello successivo (simulando la contrazione del tensore). I parametri sono
esposti come `ParameterVector` per uso in circuiti variationali.
"""
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def build_ttn_circuit(n_qubits: int, reps: int = 1, name: str = "ttn") -> QuantumCircuit:
    """Costruisce un circuito TTN binario su `n_qubits`.

    Parametri:
    - n_qubits: numero di qubit fisici disponibili
    - reps: numero di ripetizioni/strati del blocco (profondità gerarchica)
    - name: nome del circuito

    Ritorna:
    - `QuantumCircuit` con parametri `theta` (ParameterVector)
    """
    if n_qubits < 1:
        raise ValueError("n_qubits deve essere >= 1")

    qc = QuantumCircuit(n_qubits, name=name)

    # Calcolo approssimativo del numero di parametri necessari
    total_params = 0
    active = n_qubits
    for _ in range(reps):
        total_params += active * 2  # due rotazioni locali per qubit: ry, rz
        total_params += (active // 2) * 1  # un parametro per il blocco entangliante a coppia
        active = (active + 1) // 2

    params = ParameterVector("theta", length=total_params)
    idx = 0

    active_qubits = list(range(n_qubits))

    for _ in range(reps):
        # Rotazioni locali su ogni qubit attivo
        for q in active_qubits:
            qc.ry(params[idx], q)
            idx += 1
            qc.rz(params[idx], q)
            idx += 1

        # Blocchi entanglianti a coppie e compressione (si conserva il primo qubit della coppia)
        next_active = []
        for i in range(0, len(active_qubits) - 1, 2):
            q1 = active_qubits[i]
            q2 = active_qubits[i + 1]
            qc.cx(q1, q2)
            qc.ry(params[idx], q2)
            idx += 1
            qc.cx(q1, q2)
            next_active.append(q1)

        # Se c'è un qubit dispari lo portiamo al livello successivo senza modificarlo
        if len(active_qubits) % 2 == 1:
            next_active.append(active_qubits[-1])

        active_qubits = next_active

        # Se siamo ridotti a un singolo qubit possiamo fermarci
        if len(active_qubits) <= 1:
            break

    return qc


if __name__ == "__main__":
    # Esempio rapido: costruisce un TTN su 8 qubit con 3 ripetizioni e stampa il circuito
    qc = build_ttn_circuit(8, reps=3)
    print(qc.draw(output='text'))
