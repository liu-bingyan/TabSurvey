In the TabSurvey code.

The timer returns the average training time for each fold.

My GPU has 5.437 TFLOPS for FP32.
 - Given most of the case the utilizaiton rate should be 40%, let us underestimate it by 30% and regard it as 1.6TFLOPS.
My CPU has 6 cores and 2.6 GHz. shoud have 48 GFLOPS.

Test MLP on adult on cuda, the best parameters are 
MLP: {hidden_dim: 47, n_layers: 4, learning_rate: 8.4e-4}
Adult consists of (32561, 14) data
Thus the number of connections are 14*47+3*47*47+47*1 = 7332
Thus the operations per epoch are 6*7332*32561 = 1432423512 = 1.4G. 
         This is overestimating since there is a train and test split and training set are split into 5 pieces. 
Thus the total operations should be 0.5T FLOPS.

Thus the training should only take 0.3 seconds. But now it takes 80 seconds. 

This computation is underestimating the workload because the categorical features.

In code written by myself (docker container):

1. record number of epochs actually executed for more accurate computation --- ***low priority*** this will over-estimate the actual utilization. Despite that the utilization is still very low.
2. according to line_profiler, most of the time is spend on to_device. Improve this.
