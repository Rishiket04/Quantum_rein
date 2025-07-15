# Optimizing Quantum Parameters using Reinforcement Learning

This repository implements a hybrid approach for optimizing quantum circuit parameters using Proximal Policy Optimization (PPO) combined with gradient-based fine-tuning. The project targets a three-qubit parameterized quantum circuit with 30 continuous parameters, maximizing fidelity against a random target quantum state.

## Features

* **Quantum Circuit Simulation**: Built with Qiskit, supports parameterized single- and two-qubit gates.
* **PPO Reinforcement Learning**: Multi-agent PPO framework using PyTorch, with clipped surrogate objective and entropy regularization.
* **Gradient-based Fine-Tuning**: Finite-difference gradient estimation and noise injection to escape local optima.
* **Ensemble Strategy**: Maintains multiple high-performing parameter sets for robustness.
* **Experience Replay**: Stores high-fidelity experiences to reinforce effective policies.
* **Visualization**: Training history plots for rewards and fidelities using Matplotlib.

## Table of Contents

* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Configuration](#configuration)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/quantum-rl-optimization.git
   cd quantum-rl-optimization
   ```
2. (Optional) Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run Training**: Launch PPO training and fine-tuning:

   ```bash
   python main.py --epochs 150 --agents 20 --ensemble-size 5
   ```
2. **Monitor Progress**: View printed metrics every 10 epochs and training history saved to `training_history.png`.
3. **Load Saved Parameters**: After training, best parameters are saved to `best_params.npy`.
4. **Evaluate Fidelity**: Use:

   ```bash
   python evaluate.py --params best_params.npy
   ```

## Project Structure

```
├── src
│   ├── circuit.py          # Builds parameterized 3-qubit circuit and fidelity eval
│   ├── models.py           # Policy and Value network definitions
│   ├── ppo_trainer.py      # PPO training loop and experience memory
│   ├── fine_tune.py        # Gradient-based fine-tuning implementation
│   ├── utils.py            # Helper functions (e.g., normalization, logging)
│   └── main.py             # Entry point: orchestrates training and fine-tuning
├── requirements.txt        # Python dependencies
├── training_history.png    # Example output plot
├── best_params.npy         # Saved best parameter vector
└── README.md
```

## Configuration

Key hyperparameters can be adjusted in `main.py` or passed as arguments:

* `--epochs`: Total PPO epochs (default: 150)
* `--agents`: Number of parallel agents (default: 20)
* `--batch-size`: Total steps per epoch (default: 320)
* `--ensemble-size`: Number of fine-tuned parameter sets (default: 5)
* `--lr-policy`, `--lr-value`: Learning rates for policy and value networks

## Results

* **PPO Best Fidelity**: \~0.7279 after 52 epochs
* **Fine-Tuned Fidelity**: Near-perfect (0.999999)
* **Average Fidelity**: >0.97 across ensemble members

Example plots and detailed results are discussed in the project report.

## Contributing

Contributions are welcome! Please fork the repository and open a pull request with improvements or bug fixes.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
