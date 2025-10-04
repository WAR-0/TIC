#!/bin/bash
# TIC Parameter Recovery v5 Runner
# Evidence-aligned priors with Ex-Gaussian noise model

# Default values
N_SIM=${N_SIM:-200}
N_PARTICIPANTS=${N_PARTICIPANTS:-35}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-sim)
            N_SIM="$2"
            shift 2
            ;;
        --n-participants)
            N_PARTICIPANTS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --n-sim N              Number of simulations (default: 200)"
            echo "  --n-participants N     Participants per simulation (default: 35)"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --n-sim 500 --n-participants 35"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Running TIC Parameter Recovery v5"
echo "=================================="
echo "Simulations: $N_SIM"
echo "Participants: $N_PARTICIPANTS"
echo ""

N_SIM=$N_SIM N_PARTICIPANTS=$N_PARTICIPANTS python3 "$(dirname "$0")/simulations/parameter_recovery_v5_final.py"
