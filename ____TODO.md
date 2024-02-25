# cloud device
- learn how to use cloud computing resources

# understand the code
- figure out how to use GPUS while training on Windows
- figure out the "On Device : " label about where it come from?
- where does the "device:cuda" come from and why it looks like this in __bug_log_gpu
- where does the epochs play an role?  Didn't see that in the code
## urgent
- what does the time mean? mean each epoch or each fold? i assume it is fold
- understand the strucutre of the MLP and SAINT and TabTrasform
- understand the adult data and covertype. Where does 7 come from?

# verify the resulst
- run the test on docker see its accuracy and AUC.
- how long it actually takes? Is it reasonable
- run the mlp_num
## time issue
currently the time for each epoch is 13 (1 epoch in total) and 18.16 (5 epochs in total). Ben's machine has 4.7 GFLOPS (training using GPUs) Ben tested 400 epochs and takes 8-12 hours for 5 fold validation, 1.6 - 2.4 hours per fold, each epoch takes 14 seconds to 21 seconds. It matches with my machine using CPUs. 
- estimate how long the training of MLP on covertype should take
- charles estimated that it should not take longer than an hour
### solution
for now we set the dataset to 10000 records

# the model issue
- the bins optimization in mlp_num does not work, all bins converge to a large bin
- it works well adults, using the same code.
- figuring out what is happening:
    -- if the code is correct 
    -- if deleting some features helps
    -- another way to write MLP_NUM
### Solutions
- Add M-matrix
    - it seems works
- Debugging the model
    - dataitself: class-sentitive learning
    - initialize in a random way
    - test on artificial data see if it convergages

Feb 06 

# the model issue
the bins are not converging well 
some bins are off the range 01
why using max-min
why not using random initialization

1. bins are overlaping 
    - mean is close to offset
    - width is approx 2

    - bigger width means the transformation is approximately linear
    - small w means the transformed feature is approximately 0-1

    - linear relations (y as function of the particular feature) should have overlapping bins
        - we can test it on artifical data 
    
        -    f = 100 x1 + 0.01 x2 + epsilon
    - non-linear relations should have distinct bins
        -   f  = sum of indicators
    - mixture should have mixed behavior
        -   f = sin(x) for x > 0 and 

2. the width 
    - when overlapping, what does the width mean? 
    - does very wide width indiciate irrelavent feature? or important features? 




GFLOPS float32
softmax