#!/bin/bash
export OMP_NUM_THREADS=1
python run_pattern_alignment_cohort2.py --iteration 1 --k 30 &&
python run_pattern_alignment_cohort2.py --iteration 3 --k 30 &&
python run_pattern_alignment_cohort2.py --iteration 4 --k 30 &&
python run_pattern_alignment_cohort2.py --iteration 5 --k 30 &&
python run_pattern_alignment_cohort2.py --iteration 1 --k 100 &&
python run_pattern_alignment_cohort2.py --iteration 2 --k 100 &&
python run_pattern_alignment_cohort2.py --iteration 3 --k 100 &&
python run_pattern_alignment_cohort2.py --iteration 4 --k 100 &&
python run_pattern_alignment_cohort2.py --iteration 5 --k 100 