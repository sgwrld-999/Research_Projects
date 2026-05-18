#!/bin/bash

################################################################################
# Ablation Study Runner for Linformer-IDS
#
# This script automates the execution of ablation studies for the Linformer-IDS
# intrusion detection system. It runs comprehensive experiments to evaluate
# architectural components, design choices, and efficiency metrics.
#
# Usage:
#   ./run_ablation_studies.sh [study] [device] [data]
#
# Arguments:
#   study   - Type of ablation: architecture, design, efficiency, all, or test
#             (default: all)
#   device  - Computation device: cuda or cpu (default: cpu)
#   data    - Path to CIC-IoT training data
#             (default: lightweight-transformer-ids/data/raw/ciciot_2023/ciciot_training_50_50.csv)
#
# Examples:
#   ./run_ablation_studies.sh                          # Run all studies on CPU
#   ./run_ablation_studies.sh architecture cuda        # Architecture study on GPU
#   ./run_ablation_studies.sh design cpu /path/to/data # Design study with custom data
#   ./run_ablation_studies.sh test cpu                 # Quick test run
#
# Output:
#   Results: results/ciciot_binary_classification/<study_name>/<timestamp>/
#   Logs:    logs/<study_name>/<timestamp>/
#
################################################################################

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
STUDY_TYPE="${1:-all}"
DEVICE="${2:-cpu}"
DATA_PATH="${3:-lightweight-transformer-ids/data/raw/ciciot_2023/ciciot_training_50_50.csv}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Timestamp for this run
RUN_TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "\n${BLUE}=================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=================================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

check_file_exists() {
    if [ ! -f "$1" ]; then
        print_error "File not found: $1"
        return 1
    fi
    return 0
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 not found. Please install Python 3.8 or later."
        exit 1
    fi
    print_success "Python3 found: $(python3 --version)"
}

run_ablation_study() {
    local study_name=$1
    local config_file=$2

    print_header "Starting $study_name Ablation Study"

    # Check if config exists
    if ! check_file_exists "$config_file"; then
        print_error "Skipping $study_name (config not found)"
        return 1
    fi

    # Check if data exists
    if ! check_file_exists "$DATA_PATH"; then
        print_error "Data file not found: $DATA_PATH"
        return 1
    fi

    print_info "Configuration: $config_file"
    print_info "Data: $DATA_PATH"
    print_info "Device: $DEVICE"
    print_info "Timestamp: $RUN_TIMESTAMP"

    echo ""

    # Run ablation study
    python3 lightweight-transformer-ids/src/run_ablation.py \
        --config "$config_file" \
        --data "$DATA_PATH" \
        --device "$DEVICE"

    if [ $? -eq 0 ]; then
        print_success "$study_name ablation study completed"
        return 0
    else
        print_error "$study_name ablation study failed"
        return 1
    fi
}

################################################################################
# Main Execution
################################################################################

main() {
    print_header "Linformer-IDS Ablation Study Runner"

    echo -e "${YELLOW}Run Configuration:${NC}"
    echo "  Study Type:  $STUDY_TYPE"
    echo "  Device:      $DEVICE"
    echo "  Data:        $DATA_PATH"
    echo "  Timestamp:   $RUN_TIMESTAMP"

    # Check Python availability
    check_python

    # Initialize result tracking
    RESULTS_DIR="results/ciciot_binary_classification"
    LOGS_DIR="logs"

    # Create necessary directories
    mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
    print_success "Directories initialized"

    # Run studies based on selection
    case "$STUDY_TYPE" in
        architecture)
            run_ablation_study "Architecture" "lightweight-transformer-ids/configs/ablation/architecture.yaml"
            ;;
        design)
            run_ablation_study "Design" "lightweight-transformer-ids/configs/ablation/design.yaml"
            ;;
        efficiency)
            run_ablation_study "Efficiency" "lightweight-transformer-ids/configs/ablation/efficiency.yaml"
            ;;
        model_comparison)
            run_ablation_study "Model_Comparison" "lightweight-transformer-ids/configs/ablation/model_comparison.yaml"
            ;;
        multiclass)
            run_ablation_study "Multiclass_Comparison" "lightweight-transformer-ids/configs/ablation/multiclass_comparison.yaml"
            ;;
        test)
            run_ablation_study "Test" "lightweight-transformer-ids/configs/ablation/test_quick.yaml"
            ;;
        all)
            print_header "Running All Ablation Studies"

            FAILED=0

            # Run architecture ablation
            run_ablation_study "Architecture" "lightweight-transformer-ids/configs/ablation/architecture.yaml"
            [ $? -ne 0 ] && ((FAILED++))

            # Run design ablation
            run_ablation_study "Design" "lightweight-transformer-ids/configs/ablation/design.yaml"
            [ $? -ne 0 ] && ((FAILED++))

            # Run efficiency ablation
            run_ablation_study "Efficiency" "lightweight-transformer-ids/configs/ablation/efficiency.yaml"
            [ $? -ne 0 ] && ((FAILED++))

            if [ $FAILED -eq 0 ]; then
                print_success "All ablation studies completed successfully"
            else
                print_error "$FAILED ablation study(ies) failed"
                exit 1
            fi
            ;;
        *)
            print_error "Unknown study type: $STUDY_TYPE"
            echo ""
            echo "Valid options:"
            echo "  architecture - Run architectural ablation (k, depth, dim, heads)"
            echo "  design       - Run design choice ablation (loss, encoding, preprocessing)"
            echo "  efficiency   - Run efficiency benchmarking"
            echo "  test         - Run quick test ablation"
            echo "  all          - Run all ablation studies (default)"
            exit 1
            ;;
    esac

    print_header "Ablation Study Summary"

    echo -e "${YELLOW}Results Location:${NC}"
    echo "  $RESULTS_DIR/$STUDY_TYPE*/*/results.json"
    echo "  $RESULTS_DIR/$STUDY_TYPE*/*/results.csv"
    echo ""
    echo -e "${YELLOW}Log Location:${NC}"
    echo "  $LOGS_DIR/$STUDY_TYPE*/*/"
    echo ""

    print_success "Ablation study execution completed at $(date)"
}

# Show help if requested
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    cat << 'HELP'
Ablation Study Runner for Linformer-IDS
======================================

USAGE:
    ./run_ablation_studies.sh [study_type] [device] [data_path]

STUDY TYPES:
    architecture    Architectural components (k, depth, dim, heads)
    design          Design choices (loss, encoding, preprocessing)
    efficiency      Efficiency benchmarking (latency, memory, FLOPs)
    test            Quick test run (minimal epochs)
    all             All studies (default)

DEVICES:
    cpu             CPU execution (default)
    cuda            GPU execution (requires CUDA)

EXAMPLES:
    # Run all studies on CPU
    ./run_ablation_studies.sh

    # Run architecture ablation on GPU
    ./run_ablation_studies.sh architecture cuda

    # Run design ablation with custom data
    ./run_ablation_studies.sh design cpu /path/to/data.csv

    # Quick test run
    ./run_ablation_studies.sh test cpu

OUTPUT:
    Results are saved to:
        results/ciciot_binary_classification/<study_name>/<timestamp>/
            ├── results.json      (Structured results)
            ├── results.csv       (Tabular format)
            └── summary.json      (Metadata)

    Logs are saved to:
        logs/<study_name>/<timestamp>/
            └── linformer_ids_<timestamp>.log

FEATURES:
    ✓ Automatic timestamping (no results overwritten)
    ✓ Config-driven experiments
    ✓ Flexible parameter sweeps and grid searches
    ✓ Comprehensive logging
    ✓ CPU and GPU support
    ✓ Custom data path support

For more information, see ABLATION_STUDY_GUIDE.md
HELP
    exit 0
fi

# Run main function
main
