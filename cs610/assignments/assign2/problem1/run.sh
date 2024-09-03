#!/bin/bash
file=$1
# Number of runs
NUM_RUNS=10

# Initialize total time
total_time=0

# Loop to run the command multiple times
for (( i=1; i<=NUM_RUNS; i++ ))
do
    echo "Run #$i"
    # Measure the time taken for the command to run
    start_time=$(date +%s%N)
    ./$file 5 test2/input
    end_time=$(date +%s%N)

    # Calculate the elapsed time in seconds (including fractions)
    elapsed_time=$(echo "scale=3; ($end_time - $start_time) / 1000000000" | bc)
    
    # Add to total time
    total_time=$(echo "scale=3; $total_time + $elapsed_time" | bc)
    
    # Output the elapsed time for the current run
    echo "Time for run #$i: $elapsed_time seconds"
done

# Calculate average time
average_time=$(echo "scale=3; $total_time / $NUM_RUNS" | bc)

# Output the average time
echo "Average time over $NUM_RUNS runs: $average_time seconds"

