# OCE-flash

A fast implementation of the OCE algorithm for multi-concept erasure.

## Usage

### Step 1: Compute the generic preservation term

Run the following command to compute the generic preservation term \(C_e\):

```bash
python compute_Ce.py
```

### Step 2: Perform multi-concept erasure and sample images

After computing \(C_e\), run:

```bash
bash generate_celeb.sh
```

This script performs multi-concept erasure and generates sampled images.


### Step 3: Evaluation

For celebrity evaluation, please refer to the [GIPHY Celebrity Detector Installation Guide](https://github.com/Shilin-LU/MACE/tree/main/metrics).
