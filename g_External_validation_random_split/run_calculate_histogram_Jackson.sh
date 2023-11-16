#!/bin/bash
export OMP_NUM_THREADS=1
python run_calculate_histogram_Jackson.py --iteration 1 --k 30 --method centroid --node_label TMECellType &&
python run_calculate_histogram_Jackson.py --iteration 2 --k 30 --method centroid --node_label  TMECellType &&
python run_calculate_histogram_Jackson.py --iteration 3 --k 30 --method centroid --node_label  TMECellType &&
python run_calculate_histogram_Jackson.py --iteration 4 --k 30 --method centroid --node_label  TMECellType &&
python run_calculate_histogram_Jackson.py --iteration 5 --k 30 --method centroid --node_label  TMECellType &&
python run_calculate_histogram_Jackson.py --iteration 1 --k 100 --method centroid --node_label  TMECellType &&
python run_calculate_histogram_Jackson.py --iteration 2 --k 100 --method centroid --node_label  TMECellType &&
python run_calculate_histogram_Jackson.py --iteration 3 --k 100 --method centroid --node_label  TMECellType &&
python run_calculate_histogram_Jackson.py --iteration 4 --k 100 --method centroid --node_label  TMECellType &&
python run_calculate_histogram_Jackson.py --iteration 5 --k 100 --method centroid --node_label  TMECellType &&
python run_calculate_histogram_Jackson.py --iteration 1 --k 200 --method centroid --node_label  TMECellType &&
python run_calculate_histogram_Jackson.py --iteration 2 --k 200 --method centroid --node_label  TMECellType &&
python run_calculate_histogram_Jackson.py --iteration 3 --k 200 --method centroid --node_label  TMECellType &&
python run_calculate_histogram_Jackson.py --iteration 4 --k 200 --method centroid --node_label  TMECellType &&
python run_calculate_histogram_Jackson.py --iteration 5 --k 200 --method centroid --node_label  TMECellType &&
python run_calculate_histogram_Jackson.py --iteration 1 --k 500 --method centroid --node_label  TMECellType &&
python run_calculate_histogram_Jackson.py --iteration 2 --k 500 --method centroid --node_label  TMECellType &&
python run_calculate_histogram_Jackson.py --iteration 3 --k 500 --method centroid --node_label  TMECellType &&
python run_calculate_histogram_Jackson.py --iteration 4 --k 500 --method centroid --node_label  TMECellType &&
python run_calculate_histogram_Jackson.py --iteration 5 --k 500 --method centroid --node_label  TMECellType 