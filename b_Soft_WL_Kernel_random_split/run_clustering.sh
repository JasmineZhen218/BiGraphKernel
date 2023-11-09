#!/bin/bash
export OMP_NUM_THREADS=1
python run_clustering.py --iteration 1 --k 30 &&
python run_clustering.py --iteration 2 --k 30 &&
python run_clustering.py --iteration 3 --k 30 &&
python run_clustering.py --iteration 4 --k 30 &&
python run_clustering.py --iteration 5 --k 30 &&
python run_clustering.py --iteration 1 --k 200 &&
python run_clustering.py --iteration 2 --k 200 &&
python run_clustering.py --iteration 3 --k 200 &&
python run_clustering.py --iteration 4 --k 200 &&
python run_clustering.py --iteration 5 --k 200 &&
python run_clustering.py --iteration 1 --k 500 &&
python run_clustering.py --iteration 2 --k 500 &&
python run_clustering.py --iteration 3 --k 500 &&
python run_clustering.py --iteration 4 --k 500 &&
python run_clustering.py --iteration 5 --k 500 