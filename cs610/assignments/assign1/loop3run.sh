#!/bin/bash

# Array of BLOCK values
block_values=(4 8 12 16 20 24 28 32 )

# Output file
output_file="benchmark_results.csv"

# Create/clear the output file and add header
echo "BLOCKi,BLOCKj,BLOCKk,AverageTime" > $output_file

# Function to run the benchmark
run_benchmark() {
    local blocki=$1
    local blockj=$1
    local blockk=$1

    # Update the BLOCK values in the code
    sed -i "s/^#define BLOCKi.*/#define BLOCKi ($blocki)/" assgn1.cpp
    sed -i "s/^#define BLOCKj.*/#define BLOCKj ($blockj)/" assgn1.cpp
    sed -i "s/^#define BLOCKk.*/#define BLOCKk ($blockk)/" assgn1.cpp

    # Compile the code
    g++ -O3 -o matmul assgn1.cpp -lpapi

    # Initialize sum of times
    total_time=0

    # Run 3 times and calculate the total time
    for i in {1..5}
    do
        # Capture the output
        export PAPI_OUTPUT_DIRECTORY=papi_outputs/papi_${blocki}_${blockj}_${blockk}
        mkdir -p papi_outputs/papi_${blocki}_${blockj}_${blockk}
        output=$(./matmul)

        # Extract the time from the output using grep and awk
        time=$(echo "$output" | grep "Time with blocking" | awk '{print $5}')

        # Add the time to total_time
        total_time=$((total_time + time))
    done

    # Calculate the average time
    average_time=$(echo "scale=2; $total_time / 5" | bc)

    # Print the average time
    echo "Average time with BLOCKi = $blocki, BLOCKj = $blockj, BLOCKk = $blockk: $average_time us"
    echo "----------------------------------"

    # Append result to the output file
    echo "$blocki,$blockj,$blockk,$average_time" >> $output_file
}

# Loop over each combination of BLOCK values
for blocki in "${block_values[@]}"
do
    run_benchmark $blocki $blockj $blockk
done

# Sort results by average time and save to a separate file
sort -t',' -k4 -n $output_file > sorted_results.csv

echo "Benchmark completed. Results saved in $output_file and sorted_results.csv"
