# Comparative Analysis of RL Algorithms for Maize Irrigation Optimization

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Compatible-orange)](https://gymnasium.farama.org/)

A comparative study implementing **PPO** and **A2C** reinforcement learning algorithms for optimizing water use in maize irrigation. This project extends the base work by [Alkaff et al.](https://www.mdpi.com/2227-7090/13/4/595) with a **CPU-only A2C implementation** and comprehensive performance analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Algorithms](#algorithms)
- [Performance Comparison](#performance-comparison)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)
- [References](#references)

## 🌾 Overview

This project addresses the critical challenge of sustainable water management in agriculture through reinforcement learning. By 2050, the world will need a 70% increase in food production, while irrigation already accounts for 70% of global freshwater withdrawals. 

We integrate RL algorithms with **AquaCrop-OSPy** simulations in the Gymnasium framework to develop adaptive irrigation policies that balance crop productivity with water conservation.

### Key Features

- **Two RL Implementations**: PPO (base paper) and A2C (our contribution)
- **Resource-Efficient**: A2C trained entirely on CPU with no GPU required
- **Sparse Reward Environment**: Handles delayed rewards typical in agricultural tasks
- **High-Fidelity Simulation**: Uses AquaCrop-OSPy 3.0.9 for crop growth modeling
- **Comprehensive Evaluation**: Tested against 5+ baseline irrigation strategies
- **37-Year Historical Data**: Training and evaluation using Champion, Nebraska weather data (1982-2018)
- **Practical Deployment**: Lower computational requirements suitable for real-world adoption

## 🏆 Key Results

### Comprehensive Performance Comparison

| Strategy | Yield (t/ha) | Irrigation (mm) | WE (kg/ha/mm) | Profit (USD/ha) |
|----------|--------------|-----------------|---------------|-----------------|
| **PPO** | 13.80 | 179.83 | 76.76 | 576.91 |
| **A2C** | **13.52** | **135.5** | **99.78** | **544.83** |
| Thresholds | 13.95 | 255.00 | 54.72 | 528.83 |
| Net | 13.98 | 305.61 | 45.73 | 482.21 |
| Interval | 13.81 | 380.17 | 36.32 | 377.42 |
| Random | 14.02 | 1640.83 | 8.54 | -845.60 |
| Rainfed | 8.88 | 0.0 | N/A | -130.19 |

### A2C Achievement Highlights

✅ **Highest Water Efficiency**: 99.78 kg/ha/mm (+82% vs Thresholds, +30% vs PPO)  
✅ **Lowest Water Use**: 135.5 mm (47% reduction vs Thresholds, 25% reduction vs PPO)  
✅ **Competitive Yield**: 13.52 t/ha (only 2% less than PPO)  
✅ **Strong Profitability**: $544.83/ha (3% improvement vs Thresholds)  
✅ **CPU-Only Training**: Achieved with significantly fewer computational resources

### Algorithm Trade-offs

- **PPO**: Maximises profitability ($576.91/ha) and yield with moderate water use
- **A2C**: Achieves **supreme water efficiency** (99.78 kg/ha/mm) with minimal yield trade-off
- **Resource Efficiency**: A2C trained using only CPU resources vs GPU-intensive alternatives

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/alkaffulm/aquacropgymnasium.git
cd aquacropgymnasium
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using Poetry:
```bash
poetry install
```

### Required Packages

- `gymnasium` - RL environment framework
- `stable-baselines3` - RL algorithms (PPO, A2C)
- `aquacrop` - AquaCrop-OSPy simulation
- `numpy`, `pandas` - Data processing
- `matplotlib` - Visualization
- `optuna` - Hyperparameter optimization

## 📁 Project Structure

```
aquacropgymnasium/
├── aquacropgymnasium/        # Main package directory
│   └── [environment code]
├── A2C_Results/              # A2C evaluation outputs
├── Train_Output/             # Training logs and checkpoints
├── base_paper_results/       # PPO baseline results
├── weather_data/             # Historical weather data (1982-2018)
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── optimize_hyperparameters.py  # Hyperparameter tuning
├── optimize_thresholds.py    # SMT baseline optimization
├── README.md                 # This file
├── poetry.lock              # Dependency lock file
└── pyproject.toml           # Project configuration
```

## 💻 Usage

### Training

Train the A2C agent:
```bash
python train.py --algorithm a2c --timesteps 2500000
```

Train the PPO agent:
```bash
python train.py --algorithm ppo --timesteps 2500000
```

### Evaluation

Evaluate trained models:
```bash
python evaluate.py --model_path models/a2c_model.zip --episodes 100
```

### Hyperparameter Optimization

Optimize hyperparameters using Optuna:
```bash
python optimize_hyperparameters.py --algorithm a2c --trials 50
```

### Threshold Optimization

Optimize baseline soil moisture thresholds:
```bash
python optimize_thresholds.py --trials 100
```

## 🧠 Algorithms

### Proximal Policy Optimization (PPO)

**Base Paper Implementation**

PPO uses a clipped surrogate objective to ensure stable policy updates:

```
L^CLIP(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
```

**Key Hyperparameters:**
- Learning rate: 6.34×10⁻⁴
- Steps per update: 2048
- Batch size: 512
- Epochs: 23
- Discount factor: 0.98
- Clip range: 0.22

### Advantage Actor-Critic (A2C)

**Our Implementation**

A2C maintains actor and critic networks, computing advantages as:

```
A_t = R_t + γV(s_{t+1}) - V(s_t)
```

**Key Hyperparameters:**
- Learning rate: 7×10⁻⁴
- n_steps: 5
- Discount factor: 0.99
- Entropy coefficient: 0.001
- Network: 2-layer MLP (64 neurons each)
- Training timesteps: 2,500,000
- Parallel environments: 8 (DummyVecEnv)
- **Device: CPU only** (no GPU required)

**Computational Efficiency:**
- ✅ Trained entirely on CPU resources
- ✅ Lower memory footprint than PPO
- ✅ Faster per-step updates (simpler architecture)
- ✅ Fewer hyperparameters to tune
- ✅ Suitable for resource-constrained deployments

## 📊 Performance Comparison

### Final Results Summary

Our A2C implementation achieved remarkable water efficiency with minimal computational resources:

- **Best Water Efficiency**: 99.78 kg/ha/mm (highest among all strategies)
- **Minimal Water Use**: 135.5 mm irrigation (lowest viable amount)
- **Strong Economics**: $544.83/ha profit (outperforms all except PPO)
- **Sustainable Yield**: 13.52 t/ha (competitive with all strategies)
- **CPU-Only Training**: No GPU required, making it accessible for wider adoption

### Irrigation Strategies Evaluated

1. **Reinforcement Learning:**
   - PPO (base paper)
   - A2C (our implementation)

2. **Optimized Baseline:**
   - Soil Moisture Threshold (SMT) with Optuna tuning

3. **Conventional Methods:**
   - Interval-based (7-day schedule)
   - Net Irrigation (70% TAW maintenance)
   - Rainfed (no irrigation)
   - Random (baseline control)

### Reward Mechanism

The environment uses a novel reward function balancing water conservation and yield:

```python
# Step penalty for water use
Penalty_t = Σ(I_k) if I_t > 0 else 0

# End-of-season reward
FinalReward = (DryYield)^4

# Total episode reward
TotalReward = -Σ(Penalty_t) + (DryYield)^4
```

### State Space (26 dimensions)

- Crop parameters: age, canopy cover, biomass growth, soil water depletion
- Weather data: precipitation, temperature summaries
- Total available water (TAW)

### Action Space

Binary decision at each timestep:
- **Action 0**: No irrigation (0 mm)
- **Action 1**: Apply irrigation (25 mm)

## 👥 Authors

**Syed Muhammad Shah** - [omgitsshahg@gmail.com](mailto:omgitsshahg@gmail.com)  
**Ayan Asim** - [i222139@nu.edu.pk](mailto:i222139@nu.edu.pk)  
**Abdullah Butt** - [i220591@nu.edu.pk](mailto:i220591@nu.edu.pk)  
**Asadullah Khan** - [i220589@nu.edu.pk](mailto:i220589@nu.edu.pk)

**Department of Computer Science**  
FAST National University of Computer and Emerging Sciences  
Islamabad, Pakistan

## 🙏 Acknowledgments

This work builds upon the foundation established by:

- **Base Paper**: [Alkaff et al. (2025) - "Optimizing Water Use in Maize Irrigation with Reinforcement Learning"](https://www.mdpi.com/2227-7090/13/4/595)
- **AquaCrop-Gym Framework**: Kelly et al. (2024)
- **AquaCrop-OSPy**: FAO crop simulation model
- **Stable-Baselines3**: RL algorithm implementations

## 📚 References

1. Alkaff et al., "Optimizing Water Use in Maize Irrigation with Reinforcement Learning," *Mathematics*, 2025.
2. Kelly, J., Nazari, M., and Farahmand, A., "AquaCrop-Gym: A Reinforcement Learning Environment for Crop Management," 2024.
3. Schulman, J. et al., "Proximal Policy Optimization Algorithms," *arXiv preprint arXiv:1707.06347*, 2017.
4. Mnih, V. et al., "Asynchronous Methods for Deep Reinforcement Learning," *ICML*, 2016.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or collaboration opportunities, please contact:
- Abdullah: abdullah03467496@gmail.com
- Project Repository: [https://github.com/AbdullahButt-00/irrigation-optimization-ppo-a2c](https://github.com/AbdullahButt-00/irrigation-optimization-ppo-a2c)
- Base Paper Repository: [https://github.com/alkaffulm/aquacropgymnasium](https://github.com/alkaffulm/aquacropgymnasium)

## 🌟 Citation

If you use this work in your research, please cite:

```bibtex
@misc{shah2025rl-maize-irrigation,
  title={Comparative Analysis of Reinforcement Learning Algorithms for Optimizing Water Use in Maize Irrigation: PPO and A2C Approaches},
  author={Shah, Syed Muhammad and Asim, Ayan and Abdullah, Muhammad and Khan, Asadullah},
  year={2025},
  institution={FAST National University of Computer and Emerging Sciences}
}
```

---

**Keywords**: Reinforcement Learning, PPO, A2C, Agricultural Optimization, Water Conservation, Maize Irrigation, Sparse Rewards, Deep Learning, Sustainable Agriculture
