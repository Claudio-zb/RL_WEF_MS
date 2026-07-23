# Intelligent Forecasting and Control for a Water–Energy–Food Microgrid

This repository contains the code developed for my M.Sc. thesis in Electrical Engineering at Universidad de Chile. The project combines time-series forecasting and Reinforcement Learning to support the operation of an integrated water–energy–food microgrid.

## Project Overview

Integrated water–energy–food systems involve interconnected variables, operational constraints and uncertain environmental conditions. Their efficient operation requires both anticipating future conditions and making coordinated control decisions.

This project addresses these challenges through:

1. **Time-series forecasting** using Long Short-Term Memory (LSTM) neural networks.
2. **Intelligent control** using Reinforcement Learning agents.
3. **Simulation and evaluation** under different operating scenarios.
4. **Comparison of control strategies** using quantitative performance metrics.

## Forecasting Models

The forecasting stage considers meteorological and energy variables such as:

- Solar irradiance
- Relative humidity
- Evapotranspiration
- Electrical demand

The models were implemented in Python using PyTorch. The workflow includes:

- Data cleaning and preprocessing
- Sequence generation
- Model training
- Validation and testing
- Performance evaluation
- Visualization of predictions

## Reinforcement Learning

The control stage evaluates different Reinforcement Learning algorithms, including:

- Proximal Policy Optimization (PPO)
- Deep Q-Network (DQN)
- Soft Actor-Critic (SAC)

The agents interact with a simulated environment representing the operational dynamics of the microgrid.

The development process includes:

- Definition of state and action spaces
- Reward function design
- Scenario-based simulation
- Agent training
- Policy evaluation
- Comparison of control strategies

## Technologies

- Python
- PyTorch
- Pandas
- NumPy
- Matplotlib
- LSTM neural networks
- Reinforcement Learning
- PPO
- DQN
- SAC

## Usage


### Evaluate a trained model

```bash
python src/ewms_comparison.py
```

> Replace these commands with the actual script names used in the repository.

## Evaluation

The project evaluates:

- Forecasting accuracy of the LSTM models
- Learning stability of the Reinforcement Learning agents
- Operational performance under different scenarios
- Differences between PPO, DQN and SAC
- Trade-offs between energy use, water management and agricultural requirements

Experiment outputs, figures and metrics should be stored in the `results/` directory.

## Reproducibility

To improve reproducibility, the repository should include:

- Fixed random seeds
- Model hyperparameters
- Training configuration files
- Dependency versions
- Saved experiment results
- A clear separation between raw and processed data

Some datasets may not be publicly distributed due to licensing or confidentiality restrictions. In those cases, this repository should include instructions describing the expected data format.

## Related Publication

This research is associated with the following publication:

**A novel sustainable approach of reinforcement learning control for a water-energy-food microgrid**

IEEE Latin American Conference on Computational Intelligence, 2025.

## Author

**Claudio Zúñiga Bauerle**

Electrical Engineer and M.Sc. in Electrical Engineering  
Universidad de Chile

- [LinkedIn](https://www.linkedin.com/in/claudio-z%C3%BA%C3%B1iga-bauerle-63303a220)
- Email: claudio.zuniga.b@ug.uchile.cl

## Acknowledgements

This work was developed as part of my M.Sc. thesis at Universidad de Chile, with contributions from the associated research team and academic supervisors.

