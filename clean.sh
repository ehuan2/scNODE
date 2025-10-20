# Default: don't clean tensorboard
clean_tb=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --clean-tb)
            clean_tb=true
            shift
            ;;
        --no-clean-tb)
            clean_tb=false
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--clean-tb | --no-clean-tb]"
            exit 1
            ;;
    esac
done

# --- Main cleaning steps ---
echo "🧹 Cleaning project directories..."

# Example: remove checkpoints, logs, temp files, etc.
rm logs/logging/*
rm nohup*
rm *.log
rm logs/*.out

# --- TensorBoard cleaning (conditional) ---
if [ "$clean_tb" = true ]; then
    echo "🧼 Cleaning TensorBoard logs..."
    rm -rf logs/scNODE_runs/
else
    echo "⚠️  Skipping TensorBoard log cleanup."
fi

echo "✅ Done."