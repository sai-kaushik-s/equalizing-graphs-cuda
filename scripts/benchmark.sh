#!/bin/bash

# Configuration: Adjust these ranges as needed
N_VALUES=(1000 5000 10000 25000 50000 100000 250000 500000 750000 1000000 2000000 3000000)
K_VALUES=(4 16 24 32 48 64 96 128)
T_VALUES=(5 10 25 50)
CSV_FILE="benchmark_results.csv"

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m'

# Ensure directories exist
mkdir -p input/int input/float output/knn output/approx_knn output/kmeans

# Write CSV Header
echo "n,k,t,algo,precision,gpu_time,cpu_time,mae" > $CSV_FILE

for n in "${N_VALUES[@]}"; do
    for k in "${K_VALUES[@]}"; do
        echo -e "${GREEN}--------------------------------------------------${NC}"
        echo -e "${GREEN}Benchmarking Group: n=$n, k=$k${NC}"
        echo -e "${GREEN}--------------------------------------------------${NC}"

        # 1. Generate Input Data once for this (n, k)
        echo "Generating base input files (n=$n, k=$k)..."
        python3 scripts/generateInput.py --n $n --k $k --T 1 > /dev/null
        
        INT_INPUT="input/int/dataset_n_${n}_k_${k}_T_1.txt"
        FLOAT_INPUT="input/float/dataset_n_${n}_k_${k}_T_1.txt"

        if [[ ! -f "$INT_INPUT" ]]; then
            echo "Error: Base input file not generated correctly."
            continue
        fi

        # 2. Run T-independent algorithms: KNN and Approx KNN
        # KNN (Baseline)
        echo "Running knn (baseline)..."
        KNN_OUT=$(./bin/runAlgo "$INT_INPUT" "$FLOAT_INPUT" "knn")
        
        GPU_TIME_KNN_INT=$(echo "$KNN_OUT" | grep "GPU Time (INT32):" | cut -d':' -f2 | xargs)
        CPU_TIME_KNN_INT=$(echo "$KNN_OUT" | grep "CPU Time (INT32):" | cut -d':' -f2 | xargs)
        GPU_TIME_KNN_FLOAT=$(echo "$KNN_OUT" | grep "GPU Time (FLOAT):" | cut -d':' -f2 | xargs)
        CPU_TIME_KNN_FLOAT=$(echo "$KNN_OUT" | grep "CPU Time (FLOAT):" | cut -d':' -f2 | xargs)
        
        KNN_INT_FILE="output/knn/n_${n}_k_${k}_int.txt"
        KNN_FLOAT_FILE="output/knn/n_${n}_k_${k}_float.txt"

        echo "$n,$k,-,knn,int,\"$GPU_TIME_KNN_INT\",\"$CPU_TIME_KNN_INT\",0.0" >> $CSV_FILE
        echo "$n,$k,-,knn,float,\"$GPU_TIME_KNN_FLOAT\",\"$CPU_TIME_KNN_FLOAT\",0.0" >> $CSV_FILE

        # Approx KNN
        echo "Running approx_knn..."
        APPROX_OUT=$(./bin/runAlgo "$INT_INPUT" "$FLOAT_INPUT" "approx_knn")
        
        GPU_TIME_APPROX_INT=$(echo "$APPROX_OUT" | grep "GPU Time (INT32):" | cut -d':' -f2 | xargs)
        CPU_TIME_APPROX_INT=$(echo "$APPROX_OUT" | grep "CPU Time (INT32):" | cut -d':' -f2 | xargs)
        GPU_TIME_APPROX_FLOAT=$(echo "$APPROX_OUT" | grep "GPU Time (FLOAT):" | cut -d':' -f2 | xargs)
        CPU_TIME_APPROX_FLOAT=$(echo "$APPROX_OUT" | grep "CPU Time (FLOAT):" | cut -d':' -f2 | xargs)
        
        APPROX_INT_FILE="output/approx_knn/n_${n}_k_${k}_int.txt"
        APPROX_FLOAT_FILE="output/approx_knn/n_${n}_k_${k}_float.txt"
        
        MAE_APPROX_INT=$(python3 scripts/maeLoss.py "$KNN_INT_FILE" "$APPROX_INT_FILE" | grep "MAE:" | cut -d':' -f2 | xargs)
        MAE_APPROX_FLOAT=$(python3 scripts/maeLoss.py "$KNN_FLOAT_FILE" "$APPROX_FLOAT_FILE" | grep "MAE:" | cut -d':' -f2 | xargs)

        echo "$n,$k,-,approx_knn,int,\"$GPU_TIME_APPROX_INT\",\"$CPU_TIME_APPROX_INT\",$MAE_APPROX_INT" >> $CSV_FILE
        echo "$n,$k,-,approx_knn,float,\"$GPU_TIME_APPROX_FLOAT\",\"$CPU_TIME_APPROX_FLOAT\",$MAE_APPROX_FLOAT" >> $CSV_FILE

        # 3. Inner loop for T-dependent algorithm: KMeans
        for t in "${T_VALUES[@]}"; do
            echo "Running kmeans (t=$t)..."
            sed -i "3s/.*/$t/" "$INT_INPUT"
            sed -i "3s/.*/$t/" "$FLOAT_INPUT"

            KMEANS_OUT=$(./bin/runAlgo "$INT_INPUT" "$FLOAT_INPUT" "kmeans")
            
            GPU_TIME_KMEANS_INT=$(echo "$KMEANS_OUT" | grep "GPU Time (INT32):" | cut -d':' -f2 | xargs)
            CPU_TIME_KMEANS_INT=$(echo "$KMEANS_OUT" | grep "CPU Time (INT32):" | cut -d':' -f2 | xargs)
            GPU_TIME_KMEANS_FLOAT=$(echo "$KMEANS_OUT" | grep "GPU Time (FLOAT):" | cut -d':' -f2 | xargs)
            CPU_TIME_KMEANS_FLOAT=$(echo "$KMEANS_OUT" | grep "CPU Time (FLOAT):" | cut -d':' -f2 | xargs)
            
            KMEANS_INT_FILE="output/kmeans/n_${n}_k_${k}_t_${t}_int.txt"
            KMEANS_FLOAT_FILE="output/kmeans/n_${n}_k_${k}_t_${t}_float.txt"
            
            MAE_KMEANS_INT=$(python3 scripts/maeLoss.py "$KNN_INT_FILE" "$KMEANS_INT_FILE" | grep "MAE:" | cut -d':' -f2 | xargs)
            MAE_KMEANS_FLOAT=$(python3 scripts/maeLoss.py "$KNN_FLOAT_FILE" "$KMEANS_FLOAT_FILE" | grep "MAE:" | cut -d':' -f2 | xargs)

            echo "$n,$k,$t,kmeans,int,\"$GPU_TIME_KMEANS_INT\",\"$CPU_TIME_KMEANS_INT\",$MAE_KMEANS_INT" >> $CSV_FILE
            echo "$n,$k,$t,kmeans,float,\"$GPU_TIME_KMEANS_FLOAT\",\"$CPU_TIME_KMEANS_FLOAT\",$MAE_KMEANS_FLOAT" >> $CSV_FILE

            rm "$KMEANS_INT_FILE" "$KMEANS_FLOAT_FILE"
        done

        # 4. Cleanup
        echo "Cleaning up group files..."
        rm "$INT_INPUT" "$FLOAT_INPUT"
        rm -rf output/knn/* output/approx_knn/*
    done
done

echo -e "${GREEN}Benchmarking complete. Results saved to $CSV_FILE.${NC}"
