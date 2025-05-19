from qiskit import quantum_info,QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import DensityMatrix
from qiskit_aer import Aer
import math
import numpy as np
from scipy.stats import unitary_group
from concurrent.futures import ProcessPoolExecutor

qubits = 3
paramcount = 30
length = 2 ** qubits
backend = Aer.get_backend('statevector_simulator')

def get_circuit():
    circuit = QuantumCircuit(qubits)
    params = []
    for i in range(paramcount):
        params.append(Parameter('t' + str(i)))
    
    circuit.u(params[0], params[1], params[2], 0)
    circuit.u(params[3], params[4], params[5], 1)
    circuit.rxx(params[6], 0, 1)
    circuit.u(params[7], params[8], params[9], 0)
    circuit.u(params[10], params[11], params[12], 2)
    circuit.rxx(params[13], 0, 2)
    circuit.u(params[14], params[15], params[16], 1)
    circuit.u(params[17], params[18], params[19], 2)
    circuit.rxx(params[20], 1, 2)
    circuit.u(params[21], params[22], params[23], 0)
    circuit.u(params[24], params[25], params[26], 1)
    circuit.u(params[27], params[28], params[29], 2)

    return circuit

def perform_gs(target, circuit):
    acount = 500
    pcount = paramcount
    gravity = 100
    sgravity = gravity
    agrav = 0.1
    iter = 1000
    agents = []
    
    for i in range(acount):
        agents.append([np.random.rand(pcount) * 2 * math.pi  - math.pi, 0, np.zeros(pcount)])

    best_sol = []
    best_sol_fit = 1000
    
    for i in range(iter):
        gravity = sgravity * math.exp(-agrav * (i / iter))
        for agent in agents:
            circ = circuit.assign_parameters(agent[0])
            result = backend.run(transpile(circ, backend)).result().get_statevector()

            # minization problem of fitness value
            fitness = 1 - quantum_info.state_fidelity(target, result)
            agent[1] = fitness

            if best_sol_fit > agent[1]:
                best_sol = agent[0]
                best_sol_fit = agent[1]

        agents.sort(key=lambda x: x[1])     # after sorting, first element is best and last element is worst
        mass = []
        total = 0

        for agent in agents:
            m = (agents[acount-1][1] - agent[1]) / (agents[acount-1][1] - agents[0][1])
            total = total + m
            mass.append(m)
        
        for agent,m in zip(agents,mass):
            agent.append(m / total)

        for agent in agents:
            force = np.zeros(pcount)
            for neighbor in agents:
                dist = np.sqrt(np.sum(np.square(np.array(agent[0]) - np.array(neighbor[0]))))
                if dist == 0:
                    continue

                force = force + gravity * ((agent[3] * neighbor[3]) / (dist)) * np.random.rand(pcount) * (agent[0] - neighbor[0])
        
            nextvel = agent[2] if agent[3] == 0 else (np.random.rand() * agent[2] + force / agent[3])
            agent[2] = nextvel
            agent[0] = agent[0] + nextvel

    return best_sol

def get_values(proc):
    target = quantum_info.random_statevector(length)
    print("[{0}] Target: {1}".format(proc, target), flush=True)
    circuit = get_circuit()
     
    best_sol = perform_gs(target, circuit)
    
    circ = circuit.assign_parameters(best_sol)
    result = backend.run(transpile(circ, backend)).result().get_statevector()
    fid = quantum_info.state_fidelity(target, result)
#    print("[{0}] Result: {1}".format(proc, result), flush=True)
#    print("[{0}] Parameters: {1}".format(proc, best_sol), flush= True)
    print("[{0}] Fitness: {1}".format(proc, fid), flush=True)

    return fid

if __name__ == "__main__":
    batch = 100
    total_fid = 0

    result_list = []
    with ProcessPoolExecutor(max_workers=batch) as executor:
        for i in range(batch):
            result_list.append(executor.submit(get_values, i))

    for output in result_list:
        fid = output.result()
        total_fid += fid
       
    print("Average Fitness: ", total_fid / batch, flush=True)
