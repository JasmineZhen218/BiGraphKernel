#!/bin/bash
python run_community_detection.py --iteration 2 --PhenoGraph_k 100 --PopulationGraph_type complete_graph &&
python run_community_detection.py --iteration 2 --PhenoGraph_k 100 --PopulationGraph_type complete_graph_with_weak_edges_removed --weight_threshold_percentile 95 &&
python run_community_detection.py --iteration 2 --PhenoGraph_k 100 --PopulationGraph_type complete_graph_with_weak_edges_removed --weight_threshold_percentile 90 &&
python run_community_detection.py --iteration 2 --PhenoGraph_k 100 --PopulationGraph_type complete_graph_with_weak_edges_removed --weight_threshold_percentile 85 &&
python run_community_detection.py --iteration 2 --PhenoGraph_k 100 --PopulationGraph_type complete_graph_with_weak_edges_removed --weight_threshold_percentile 80 &&

python run_community_detection.py --iteration 2 --PhenoGraph_k 100 --PopulationGraph_type knn_graph --knn_k 3 &&
python run_community_detection.py --iteration 2 --PhenoGraph_k 100 --PopulationGraph_type knn_graph --knn_k 5 &&
python run_community_detection.py --iteration 2 --PhenoGraph_k 100 --PopulationGraph_type knn_graph --knn_k 10 &&
python run_community_detection.py --iteration 2 --PhenoGraph_k 100 --PopulationGraph_type knn_graph --knn_k 15 &&
python run_community_detection.py --iteration 2 --PhenoGraph_k 100 --PopulationGraph_type knn_graph --knn_k 20 &&
python run_community_detection.py --iteration 2 --PhenoGraph_k 100 --PopulationGraph_type knn_graph --knn_k 25 &&
python run_community_detection.py --iteration 2 --PhenoGraph_k 100 --PopulationGraph_type knn_graph --knn_k 30 &&

python run_community_detection.py --iteration 2 --PhenoGraph_k 100 --PopulationGraph_type two_step_knn_graph --knn_k 3 &&
python run_community_detection.py --iteration 2 --PhenoGraph_k 100 --PopulationGraph_type two_step_knn_graph --knn_k 5 &&
python run_community_detection.py --iteration 2 --PhenoGraph_k 100 --PopulationGraph_type two_step_knn_graph --knn_k 10 &&
python run_community_detection.py --iteration 2 --PhenoGraph_k 100 --PopulationGraph_type two_step_knn_graph --knn_k 15 &&
python run_community_detection.py --iteration 2 --PhenoGraph_k 100 --PopulationGraph_type two_step_knn_graph --knn_k 20 &&
python run_community_detection.py --iteration 2 --PhenoGraph_k 100 --PopulationGraph_type two_step_knn_graph --knn_k 25 &&
python run_community_detection.py --iteration 2 --PhenoGraph_k 100 --PopulationGraph_type two_step_knn_graph --knn_k 30
