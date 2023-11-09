ITERATION=(1 2 3 4 5)
PhenoGraph_k=(30)
SURVIVAL_TYPE=('Overall') #'Overall' 'Disease-specific' 

for ITERATION in ${ITERATION[@]};do
    for PhenoGraph_k in ${PhenoGraph_k[@]};do
        for SURVIVAL_TYPE in ${SURVIVAL_TYPE[@]};do
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type complete_graph --size_smallest_cluster 20 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type complete_graph_with_weak_edges_removed --size_smallest_cluster 20 --weight_threshold_percentile 95 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type complete_graph_with_weak_edges_removed --size_smallest_cluster 20 --weight_threshold_percentile 90 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type complete_graph_with_weak_edges_removed --size_smallest_cluster 20 --weight_threshold_percentile 85 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type complete_graph_with_weak_edges_removed --size_smallest_cluster 20 --weight_threshold_percentile 80 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type complete_graph_with_weak_edges_removed --size_smallest_cluster 20 --weight_threshold_percentile 75 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type complete_graph_with_weak_edges_removed --size_smallest_cluster 20 --weight_threshold_percentile 70 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type complete_graph_with_weak_edges_removed --size_smallest_cluster 20 --weight_threshold_percentile 65 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type complete_graph_with_weak_edges_removed --size_smallest_cluster 20 --weight_threshold_percentile 60 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type complete_graph_with_weak_edges_removed --size_smallest_cluster 20 --weight_threshold_percentile 55 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type complete_graph_with_weak_edges_removed --size_smallest_cluster 20 --weight_threshold_percentile 50 &&
            
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type knn_graph --size_smallest_cluster 20 --knn_k 3 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type knn_graph --size_smallest_cluster 20 --knn_k 5 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type knn_graph --size_smallest_cluster 20 --knn_k 10 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type knn_graph --size_smallest_cluster 20 --knn_k 15 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type knn_graph --size_smallest_cluster 20 --knn_k 20 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type knn_graph --size_smallest_cluster 20 --knn_k 25 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type knn_graph --size_smallest_cluster 20 --knn_k 30 &&

            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type two_step_knn_graph --size_smallest_cluster 20 --knn_k 3 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type two_step_knn_graph --size_smallest_cluster 20 --knn_k 5 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type two_step_knn_graph --size_smallest_cluster 20 --knn_k 10 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type two_step_knn_graph --size_smallest_cluster 20 --knn_k 15 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type two_step_knn_graph --size_smallest_cluster 20 --knn_k 20 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type two_step_knn_graph --size_smallest_cluster 20 --knn_k 25 &&
            python run_survival_analysis.py --iteration $ITERATION --PhenoGraph_k $PhenoGraph_k --survival_type "$SURVIVAL_TYPE" --PopulationGraph_type two_step_knn_graph --size_smallest_cluster 20 --knn_k 30 
        done
    done
done