This is the codebase for the paper "Prompt Mixing in Diffusion Models using the Black Scholes Algorithm".

The folder structure is as follows:

data/ - contains the python script used for data creation, and also the prompts for all 4 sets used in the paper.
models/ - contains the scripts for inference using each of the prompt mixing techniques compared with in the paper.
metrics.py - contains the evaluation script.
run_batch_*.py - these are the python scripts that are used to run the experiments for each of the prompt mixing techniques.

Note that we use the same main script (run_batch_altsamp) for Alternating Sampling as well as Step-wise sampling. Make sure to use the correct path to the model inside the script, for running each of the methods.
