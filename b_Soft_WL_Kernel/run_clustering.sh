#!/bin/bash
export OMP_NUM_THREADS=1
python run_clustering.py --iteration 1 &&
python run_clustering.py --iteration 2 &&
python run_clustering.py --iteration 3 &&
python run_clustering.py --iteration 4 &&
python run_clustering.py --iteration 5