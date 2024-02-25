$loaders = @(1, 3, 5)
$batchSizes = 8192 #@(32768, 65532,131064,262128) #@(4096, 2048, 1024, 512, 256, 128)
$shuffleOptions = @("--no-shuffle","--shuffle")

foreach ($shuffleOption in $shuffleOptions) {
    foreach ($batchSize in $batchSizes) {
        foreach ($loader in $loaders) {
            $logFileName = "timing/kernprof/gflops2_loader${loader}_batch${batchSize}${shuffleOption}.log"
            $command = "kernprof -lv __gflops2.py --num_epochs 300 --data_loader $loader --batch_size $batchSize $shuffleOption > $logFileName"
            #$command = "python __gflops2.py --data_loader $loader --batch_size $batchSize $shuffleOption"
            Invoke-Expression $command
        }
    }
}
