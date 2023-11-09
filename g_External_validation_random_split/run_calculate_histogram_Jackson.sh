#!/bin/bash
export OMP_NUM_THREADS=1
# python run_calculate_histogram_Jackson.py --iteration 1 --k 30 --method centroid &&
# python run_calculate_histogram_Jackson.py --iteration 2 --k 30 --method centroid &&
# python run_calculate_histogram_Jackson.py --iteration 3 --k 30 --method centroid &&
# python run_calculate_histogram_Jackson.py --iteration 4 --k 30 --method centroid &&
# python run_calculate_histogram_Jackson.py --iteration 5 --k 30 --method centroid &&
python run_calculate_histogram_Jackson.py --iteration 1 --k 30 --method centroid &&
python run_calculate_histogram_Jackson.py --iteration 2 --k 30 --method centroid &&
python run_calculate_histogram_Jackson.py --iteration 3 --k 30 --method centroid &&
python run_calculate_histogram_Jackson.py --iteration 4 --k 30 --method centroid &&
python run_calculate_histogram_Jackson.py --iteration 5 --k 30 