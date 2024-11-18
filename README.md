# Cell-Wave Networks (CWN) - Refactored Implementation

This repository contains a refactored implementation of the Cell-Wave Networks (CWN) paper, specifically focused on the MUTAG dataset experiment. The original implementation has been updated and simplified for modern PyTorch Geometric versions and M2 chip compatibility.

## Key Changes from Original Implementation

### Modern Framework Compatibility
- Updated codebase to work with recent versions of PyTorch Geometric (PyG)
- Optimized for compatibility with Apple M2 chip architecture
- Resolved deprecated function calls and updated API usage

### Simplified Implementation
- Focused implementation on the MUTAG dataset experiment
- Streamlined the codebase by removing auxiliary functionalities not essential for the core experiment
- Enhanced reproducibility through deterministic data loading and processing

### Notable Modifications
- Implemented deterministic data splitting and loading using PyTorch generators
- Maintained core CWN architecture while removing unused features
- Added comprehensive logging and visualization of training metrics

## Core Functionality

The implementation maintains these key components:
- Graph to cell complex conversion with ring structure detection
- Sparse CIN (Cell Interaction Network) implementation
- Training pipeline with validation and testing procedures
- Performance visualization and metric tracking

## Removed Functionality
Some features from the original implementation were removed to focus on core experimentation:
- Support for multiple datasets beyond MUTAG
- Various initialization methods for cell complexes
- Extended configuration options for model architecture
- Additional visualization tools

## Usage

The main experiment can be run through the `main.py` script, which:
1. Loads and processes the MUTAG dataset
2. Converts graphs to cell complexes with rings
3. Trains the CWN model
4. Evaluates performance and generates visualization

## Requirements

- PyTorch
- PyTorch Geometric (recent version)
- NumPy
- Matplotlib
- Other dependencies as specified in requirements.txt

## Reproducibility

The implementation ensures reproducible results through:
- Fixed random seeds
- Deterministic data splitting
- Controlled training procedure

## Citation

If you use this implementation, please cite the original CWN paper:
https://arxiv.org/abs/2106.12575

## Acknowledgments

This implementation is based on the original CWN repository [\[include link\]](https://github.com/crisbodnar/cwn.git), with modifications for modern framework compatibility and focused experimentation. 