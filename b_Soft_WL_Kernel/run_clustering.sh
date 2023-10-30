#!/bin/bash
export OMP_NUM_THREADS=1
# python run_clustering.py --iteration 1 --k 100 &&
python run_clustering.py --iteration 3 --k 100 &&
python run_clustering.py --iteration 4 --k 100 &&
python run_clustering.py --iteration 5 --k 100