conda activate torch
conda install line_profiler
kernprof -lv train_timing.py --config "config/adult.yml" --model_name "MLP" --n_trials 5 --epochs 5
