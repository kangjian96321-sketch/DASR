"""DASR-aligned sequential MART; respects the user's CUDA_VISIBLE_DEVICES."""
from dasr_aligned.continual_baselines import main

if __name__ == "__main__":
    main("mart")
