## Linear VAE Optimizer Dynamics
This repository contains the codebase to reproduce the findings of the research comparing Adam against one-pass Stochastic Gradient Descent (SGD) in training linear Variational Autoencoders (VAEs). The study investigates whether Adam shifts phase boundaries, alters posterior collapse thresholds, or changes generalization dynamics compared to the exact deterministic limits established for vanilla SGD.

## Setup and Requirements
To run the experiments, you will need Python installed along with a few numerical and machine learning libraries. Create a virtual environment and install the following libraries:
```
torch
numpy
matplotlib
tqdm
```
## How to Read and Run the Code
The entirety of the research code is contained within a single file: `AdamOpt.py`. The script is structured sequentially so that the underlying mechanics are defined before the experiments are executed.
- Global configuration: Variables at the very top define the output directory, compute device, theoretical constants, and a boolean RUN dictionary to toggle individual experiments on or off.
- Core definitions: The LinearVAE class defines the encoder/decoder architecture, while generate_data_from_SCM handles the synthetic spiked-covariance data generation.
- Universal training loop: The run_training function acts as the central engine for all experiments, accepting the optimizer type as an argument to ensure a perfectly controlled, one-pass comparison with shared initialization.
- Execution: Running the command python `AdamOpt.py` will execute the experiments toggled to True and output the resulting visualizations as PDF files to your chosen directory.

## Experiment Structure
- Experiment A (Phase-boundary invariance): The code sweeps through a list of $\beta$ penalty values, trains models using both SGD and Adam, and plots their final steady-state errors to show that both optimizers share the exact same theoretical minimum and collapse threshold.
- Experiment B (Concentration): The script executes Adam training across multiple random seeds for progressively larger network sizes $N$ and measures how the variance between different runs shrinks, confirming the system approaches a deterministic state.
- Experiment C (Preconditioner delocalization): The training loop specifically tracks the internal state of Adam's second-moment buffer and plots its coefficient of variation over time to demonstrate that the adaptive learning rates quickly become uniform across all coordinates.
- Experiment D (KL-annealing window): The code applies a hyperbolic tangent schedule to the $\beta$ parameter over time using various growth rates $\gamma$ and records the exact continuous time $t$ it takes for the generalization error to drop below a target threshold.
- Experiment E (Model mismatch): The script intentionally initializes the model with two latent dimensions against a ground-truth signal of one dimension ($M=2$, $M^\ast=1$) and plots the weights of the extra dimension to show how Adam amplifies overfitting on background noise.

## Project Base
This Project is based on the code of this repository: https://github.com/Yuma-Ichikawa/LearningDynamicsVAE

The research builds on the research of Ichikawa and Hukushima:
```
@misc{ichikawa2023learningdynamicslinearvae,
      title={Learning Dynamics in Linear VAE: Posterior Collapse Threshold, Superfluous Latent Space Pitfalls, and Speedup with KL Annealing}, 
      author={Yuma Ichikawa and Koji Hukushima},
      year={2023},
      eprint={2310.15440},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2310.15440}, 
}
```
