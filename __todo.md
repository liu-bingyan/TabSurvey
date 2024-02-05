#
Get things run 
- figure out how to use GPUS while training on Windows
- figure out


# time issue
- estimate how long the training of MLP on covertype should take
- currently the time for each epoch is 13 (1 epoch in total) and 18.16 (5 epochs in total)
- It is believed to increase as the number of epochs increases
- Ben's machine has 4.7 GFLOPS (training using GPUs)
- Ben tested 400 epochs and takes 8-12 hours for 5 fold validation, 1.6 - 2.4 hours per fold, each epoch takes 14 seconds to 21 seconds. 
- It matches with my machine using CPUs. 
- I will test the time using GPUs.
- have the estimate to see this number is correct, and why Charles is not

# the model issue
- the bins optimization in mlp_num does not work, all bins converge to a large bin
- it works well adults, using the same code.
- figuring out what is happening:
    -- if the code is correct 
    -- if deleting some features helps
    -- another way to write MLP_NUM

# miscellious
debug locally: not using conda
cloud computing