$loaders = @(1, 3, 5)
$batchSizes = @(4096, 2048, 512)
$shuffleOptions = @("--no-shuffle","--shuffle")

foreach ($shuffleOption in $shuffleOptions) {
    foreach ($batchSize in $batchSizes) {
        foreach ($loader in $loaders) {
            $logFileName = "timing/kernprof/gflops2_loader${loader}_batch${batchSize}${shuffleOption}.log"
            $command = "kernprof -lv __gflops2.py --data_loader $loader --batch_size $batchSize $shuffleOption > $logFileName"
            #$command = "python __gflops2.py --data_loader $loader --batch_size $batchSize $shuffleOption"
            Invoke-Expression $command
        }
    }
}

