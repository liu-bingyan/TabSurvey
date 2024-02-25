loaders=(1 2 3) # 1: manual, 2: pytorch, 3: fast
batchSizes=8192 # (32768 65532 131064 262128) # (4096 2048 1024 512 256 128)
shuffleOptions=("--no-shuffle" "--shuffle")

for shuffleOption in "${shuffleOptions[@]}"; do
    for batchSize in "${batchSizes[@]}"; do
        for loader in "${loaders[@]}"; do
            logFileName="log/gflops2_loader${loader}_batch${batchSize}${shuffleOption}.log"
            command="kernprof -lv gflops.py --num_epochs 300 --data_loader $loader --batch_size $batchSize $shuffleOption > $logFileName"
            # command="python __gflops2.py --data_loader $loader --batch_size $batchSize $shuffleOption"
            eval $command
        done
    done
done
