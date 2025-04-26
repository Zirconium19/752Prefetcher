#!/bin/bash

# Script to process multiple traces in ~/autodl-tmp and save prefetch files to ~/autodl-tmp/ChampSim/prefetch_file
# Then run simulations with --no-base flag, handling different file extensions
# Usage: ./process_traces.sh <trace1> <trace2> ... [options]

# Default parameters
TRACES_DIR="$HOME/autodl-tmp"
PREFETCH_DIR="$HOME/autodl-tmp/ChampSim/prefetch_file"
WARMUP_INSTRUCTIONS="10"

# Function to display usage
usage() {
    echo "Usage: $0 <trace1> <trace2> ... [options]"
    echo "Example: $0 bfs-10 pagerank-10 bc-10 [options]"
    echo "Note: For processing, trace files should have .txt.xz extension"
    echo "      For simulation, trace files should have .xz or .gz extension"
    echo "Options:"
    echo "  --warmup NUM              Number of warmup instructions in millions (default: 10)"
    echo "  --help                    Display this help message"
    exit 1
}

# Check if help is requested or no arguments provided
if [ $# -lt 1 ] || [ "$1" == "--help" ]; then
    usage
fi

# Collect trace names
TRACES=()
while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
    TRACES+=("$1")
    shift
done

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --warmup)
            WARMUP_INSTRUCTIONS="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if any traces were specified
if [ ${#TRACES[@]} -eq 0 ]; then
   echo "Error: No traces specified"
   usage
fi

# Make sure the traces directory exists
if [ ! -d "$TRACES_DIR" ]; then
   echo "Error: Traces directory '$TRACES_DIR' not found"
   exit 1
fi

# Create the prefetch directory if it doesn't exist
mkdir -p "$PREFETCH_DIR"

# Print a summary of what will be done
echo "======================================"
echo "Processing ${#TRACES[@]} traces from: $TRACES_DIR"
echo "Prefetch files will be saved to: $PREFETCH_DIR"
echo "Warmup instructions: $WARMUP_INSTRUCTIONS million"
echo "Traces to process: ${TRACES[*]}"
echo "======================================"

# Process each trace - first generate all prefetch files
for TRACE_NAME in "${TRACES[@]}"; do
   #  For processing, we need the .txt.xz format
   PROCESS_TRACE_PATH="$TRACES_DIR/${TRACE_NAME}.txt.xz"
    
   #  For the prefetch file, just use the base trace name
   PREFETCH_PATH="$PREFETCH_DIR/prefetches-${TRACE_NAME}.txt"
    
   #  Verify the trace file exists
   if [ ! -f "$PROCESS_TRACE_PATH" ]; then
       echo "Warning: Trace file '$PROCESS_TRACE_PATH' not found, skipping..."
       continue
   fi
    
   echo "======================================"
   echo "Processing trace: $TRACE_NAME"
   echo "Trace path: $PROCESS_TRACE_PATH"
   echo "Prefetch file will be saved to: $PREFETCH_PATH"
   echo "======================================"
    
   # Execute the train command with generate option
   echo "Executing: ./ml_prefetch_sim.py train $PROCESS_TRACE_PATH --generate $PREFETCH_PATH --num-prefetch-warmup-instructions $WARMUP_INSTRUCTIONS"
   ./ml_prefetch_sim.py train "$PROCESS_TRACE_PATH" --generate "$PREFETCH_PATH" --num-prefetch-warmup-instructions "$WARMUP_INSTRUCTIONS"
    
   #  Check if the command was successful
   if [ $? -eq 0 ]; then
       echo "Training and prefetch generation completed successfully for $TRACE_NAME!"
       echo "Prefetch file saved to: $PREFETCH_PATH"
   else
       echo "Error: Training or prefetch generation failed for $TRACE_NAME"
   fi
    
   echo "--------------------------------------"
done

echo "All prefetch files have been generated"
echo "======================================"
echo "Now running simulations for all traces with --no-base flag"
echo "======================================"

# Now run simulations for all traces
for TRACE_NAME in "${TRACES[@]}"; do
    # Check for both .xz and .gz formats for the simulation
    SIM_TRACE_PATH_XZ="$TRACES_DIR/${TRACE_NAME}.trace.xz"
    SIM_TRACE_PATH_GZ="$TRACES_DIR/${TRACE_NAME}.trace.gz"
    PREFETCH_PATH="$PREFETCH_DIR/prefetches-${TRACE_NAME}.txt"
    
    # Determine which format to use
    if [ -f "$SIM_TRACE_PATH_XZ" ]; then
        SIM_TRACE_PATH="$SIM_TRACE_PATH_XZ"
    elif [ -f "$SIM_TRACE_PATH_GZ" ]; then
        SIM_TRACE_PATH="$SIM_TRACE_PATH_GZ"
    else
        echo "Warning: Neither $SIM_TRACE_PATH_XZ nor $SIM_TRACE_PATH_GZ found, skipping simulation..."
        continue
    fi
    
    # Verify the prefetch file exists
    if [ ! -f "$PREFETCH_PATH" ]; then
        echo "Warning: Prefetch file '$PREFETCH_PATH' not found, skipping simulation..."
        continue
    fi
    
    echo "======================================"
    echo "Running simulation for trace: $TRACE_NAME"
    echo "Trace path: $SIM_TRACE_PATH"
    echo "Prefetch file: $PREFETCH_PATH"
    echo "======================================"
    
    # Execute the run command with no-base flag
    echo "Executing: ./ml_prefetch_sim.py run $SIM_TRACE_PATH --prefetch $PREFETCH_PATH --no-base"
    ./ml_prefetch_sim.py run "$SIM_TRACE_PATH" --prefetch "$PREFETCH_PATH" --no-base
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Simulation completed successfully for $TRACE_NAME!"
    else
        echo "Error: Simulation failed for $TRACE_NAME"
    fi
    
    echo "--------------------------------------"
done
