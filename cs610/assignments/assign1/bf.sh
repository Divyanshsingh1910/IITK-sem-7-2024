#!/bin/bash


# Compile the code
g++ -O3 -o matmul assgn1.cpp -lpapi

# Initialize sum of times
total_time=0

# Run 3 times and calculate the total time
for i in {1..5}
do
    # Capture the output
    export PAPI_OUTPUT_DIRECTORY=papi_outputs/papi_bf
    mkdir -p papi_outputs/papi_bf
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
