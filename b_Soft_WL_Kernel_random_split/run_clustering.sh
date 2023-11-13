#!/bin/bash
export OMP_NUM_THREADS=1
python run_clustering.py --iteration 1 --k 30 --node_label TMECellType &&
python run_clustering.py --iteration 1 --k 100 --node_label TMECellType &&
python run_clustering.py --iteration 1 --k 200 --node_label TMECellType &&
python run_clustering.py --iteration 1 --k 500 --node_label TMECellType &&

python run_clustering.py --iteration 2 --k 30 --node_label TMECellType &&
python run_clustering.py --iteration 2 --k 100 --node_label TMECellType &&
python run_clustering.py --iteration 2 --k 200 --node_label TMECellType &&
python run_clustering.py --iteration 2 --k 500 --node_label TMECellType &&

python run_clustering.py --iteration 3 --k 30 --node_label TMECellType &&
python run_clustering.py --iteration 3 --k 100 --node_label TMECellType &&
python run_clustering.py --iteration 3 --k 200 --node_label TMECellType &&
python run_clustering.py --iteration 3 --k 500 --node_label TMECellType &&

python run_clustering.py --iteration 4 --k 30 --node_label TMECellType &&
python run_clustering.py --iteration 4 --k 100 --node_label TMECellType &&
python run_clustering.py --iteration 4 --k 200 --node_label TMECellType &&
python run_clustering.py --iteration 4 --k 500 --node_label TMECellType &&

python run_clustering.py --iteration 5 --k 30 --node_label TMECellType &&
python run_clustering.py --iteration 5 --k 100 --node_label TMECellType &&
python run_clustering.py --iteration 5 --k 200 --node_label TMECellType &&
python run_clustering.py --iteration 5 --k 500 --node_label TMECellType 
