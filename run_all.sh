#!/usr/bin/env bash
# ==============================================================================
# TF-DualMethyl (model_plus) Server Background Launcher (nohup + real-time log)
# Modeled after PhyNet-RNA / PhyNet-DTA / HMB-Net `run_ubuntu.sh`
#
# Usage:
#   bash run_all.sh                         # Full re-run in background (default)
#   bash run_all.sh --gpu 0                 # Full re-run on GPU 0 in background
#   bash run_all.sh --only-ablation --gpu 0 # Ablation-only in background
#   bash run_all.sh --fg --gpu 0            # Run in foreground (optional)
#
# After running `bash run_all.sh`, you can safely close the SSH terminal!
# Watch progress anytime with:
#   tail -f tf_dualmethyl_nohup.log
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="${LOG_FILE:-tf_dualmethyl_nohup.log}"
PID_FILE="${PID_FILE:-run_all.pid}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

# Check whether we are in launcher mode (default) or worker/foreground mode (--_worker / --fg)
RUN_IN_FG=0
FORWARD_ARGS=()
GPU_ID="auto"

ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    arg="${ARGS[i]}"
    if [[ "$arg" == "--_worker" ]] || [[ "$arg" == "--fg" ]]; then
        RUN_IN_FG=1
    else
        FORWARD_ARGS+=("$arg")
        if [[ "$arg" == "--gpu" ]] && (( i+1 < ${#ARGS[@]} )); then
            GPU_ID="${ARGS[i+1]}"
        fi
    fi
done

if [[ "$RUN_IN_FG" -eq 0 ]]; then
    # Prevent accidental duplicate background runs if already running
    if [[ -f "$PID_FILE" ]]; then
        EXISTING_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
        if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
            echo "=================================================="
            echo " [WARNING] TF-DualMethyl pipeline is ALREADY running!"
            echo "   Active PID : $EXISTING_PID"
            echo "   Log File   : $LOG_FILE"
            echo " To watch progress, run:"
            echo "   tail -f $LOG_FILE"
            echo " To stop existing process first, run:"
            echo "   kill $EXISTING_PID"
            echo "=================================================="
            exit 1
        fi
    fi

    echo "=================================================="
    echo "   TF-DualMethyl 5-Fold CV & Plotting Launcher"
    echo "   Working Dir: $SCRIPT_DIR"
    echo "   GPU Device:  $GPU_ID"
    echo "   Extra Args:  ${FORWARD_ARGS[*]:-(none - full 9-model pipeline)}"
    echo "   Log File:    $LOG_FILE"
    echo "=================================================="

    # Launch this script in background worker mode via nohup (append to preserve resume logs)
    nohup bash "$SCRIPT_DIR/run_all.sh" --_worker "${FORWARD_ARGS[@]}" >> "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" > "$PID_FILE"

    echo "Process started in background. PID: $PID"
    echo "You can safely close the terminal / SSH session now."
    echo "To watch progress, run:"
    echo "  tail -f $LOG_FILE"
    echo "To check process status, run:"
    echo "  ps -p $PID"
    echo "To stop process, run:"
    echo "  kill $PID"
    echo "=================================================="
    exit 0
fi

# ==============================================================================
# Background Worker Execution (runs inside nohup and writes to $LOG_FILE)
# ==============================================================================
source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate 2>/dev/null || true

if [[ "$GPU_ID" != "auto" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi
export PYTHONUNBUFFERED=1

echo ""
echo "======================================================================"
echo " [$(date '+%Y-%m-%d %H:%M:%S')] Starting / Resuming TF-DualMethyl Pipeline"
echo " Working Directory    : $SCRIPT_DIR"
echo " Worker PID           : $$"
echo " Python Executable    : $(which $PYTHON_BIN)"
echo " CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-auto}"
$PYTHON_BIN -u -c "import torch; print(' PyTorch Version      :', torch.__version__, '| CUDA Available:', torch.cuda.is_available())"
echo "======================================================================"

# Step 1: Run Full Experiments (or Resume from Latest Checkpoint on Interruption)
echo ""
echo "[Step 1/7] [$(date '+%H:%M:%S')] Running / Resuming 5-Fold Experiments across All Datasets..."
$PYTHON_BIN -u run_all.py "${FORWARD_ARGS[@]}"

# Step 2: Ensure Ensemble Prediction Probabilities (probability.txt) Exist
echo ""
echo "[Step 2/7] [$(date '+%H:%M:%S')] Extracting/Verifying Test Set Prediction Probabilities..."
$PYTHON_BIN -u run_probability.py

# Step 3: Cross-Species Zero-Shot Transferability Evaluation
echo ""
echo "[Step 3/7] [$(date '+%H:%M:%S')] Running Cross-Species Transferability Evaluation..."
$PYTHON_BIN -u run_cross_transfer.py

# Step 4: Generate Summary Tables (table.txt, latex_tables.tex, master epoch CSV)
echo ""
echo "[Step 4/7] [$(date '+%H:%M:%S')] Generating Performance Summary Tables & Master Epoch History..."
$PYTHON_BIN -u generate_table.py

# Step 5: Generate All Vector PDF Figures (Compare, Ablation, Confidence, Heatmaps)
echo ""
echo "[Step 5/7] [$(date '+%H:%M:%S')] Generating Publication-Grade Vector PDF Figures..."
$PYTHON_BIN -u plot/plot_com_bar.py
$PYTHON_BIN -u plot/plot_ablation_curve.py
$PYTHON_BIN -u plot/plot_curve.py
$PYTHON_BIN -u plot/plot_confidence.py
$PYTHON_BIN -u plot/plot_heatmap.py
$PYTHON_BIN -u plot/plot_attn_heatmap.py

# Step 6: Generate Interpretability Figures (t-SNE, UMAP, Motif IG, Radar)
echo ""
echo "[Step 6/7] [$(date '+%H:%M:%S')] Generating Latent Manifold & Motif Interpretability Figures..."
$PYTHON_BIN -u plot/plot_tsne.py || echo "Warning: plot_tsne.py encountered an issue, continuing..."
$PYTHON_BIN -u plot/plot_umap.py || echo "Warning: plot_umap.py encountered an issue, continuing..."
$PYTHON_BIN -u plot/plot_motif.py || echo "Warning: plot_motif.py encountered an issue, continuing..."
$PYTHON_BIN -u plot/plot_radar.py || echo "Warning: plot_radar.py encountered an issue, continuing..."
$PYTHON_BIN -u -c "import sys; sys.path.insert(0, 'plot'); import prepare_paper_figures as p; p.generate_architecture_pdf(); p.generate_manifold_and_motif_pdfs()" || true

# Step 7: Package All Results, Tables, Epoch Logs & Figures into Downloadable ZIPs
echo ""
echo "[Step 7/7] [$(date '+%H:%M:%S')] Packaging All Results & Figures into Downloadable ZIP Archives..."
$PYTHON_BIN -u pack_results.py --only-pack || echo "Warning: pack_results.py encountered an issue."

rm -f "$PID_FILE"

echo ""
echo "======================================================================"
echo " [$(date '+%Y-%m-%d %H:%M:%S')] All TF-DualMethyl Server Experiments Completed!"
echo " 1. Updated Metrics Table       : result/table.txt"
echo " 2. Ready LaTeX Rows            : result/latex_tables.tex"
echo " 3. Master Per-Epoch History    : result/all_epochs_training_history.csv"
echo " 4. Updated PDF Figures         : paper_new/figures/*.pdf"
echo " 5. Downloadable Archives       : TF-DualMethyl_results_light.zip & TF-DualMethyl_results.zip"
echo "======================================================================"
