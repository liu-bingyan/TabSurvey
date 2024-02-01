Training SAINT with config/adult.yml in env torch

Namespace(config='config/adult.yml', model_name='SAINT', dataset='Adult', objective='binary', use_gpu=False, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=5, shuffle=True, seed=221, scale=True, target_encode=True, one_hot_encode=False, batch_size=128, val_batch_size=256, early_stopping_rounds=20, epochs=300, logging_period=100, num_features=14, num_classes=1, cat_idx=[1, 3, 5, 6, 7, 8, 9, 13], cat_dims=[9, 16, 7, 15, 6, 5, 2, 42])
Train model with given hyperparameters
Loading dataset Adult...
Dataset loaded!
(32561, 14)
Scaling the data...
Using dim 32 and batch size 128
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 131, in main_once
    model = model_name(parameters, args)
  File "/opt/notebooks/models/saint.py", line 61, in __init__
    self.model.transformer = nn.DataParallel(self.model.transformer, device_ids=self.args.gpu_ids)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/torch/nn/parallel/data_parallel.py", line 142, in __init__
    _check_balance(self.device_ids)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/torch/nn/parallel/data_parallel.py", line 23, in _check_balance
    dev_props = _get_devices_properties(device_ids)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/torch/_utils.py", line 491, in _get_devices_properties
    return [_get_device_attr(lambda m: m.get_device_properties(i)) for i in device_ids]
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/torch/_utils.py", line 491, in <listcomp>
    return [_get_device_attr(lambda m: m.get_device_properties(i)) for i in device_ids]
    return get_member(torch.cuda)
    return get_member(torch.cuda)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/torch/_utils.py", line 491, in <lambda>
    return [_get_device_attr(lambda m: m.get_device_properties(i)) for i in device_ids]
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/torch/cuda/__init__.py", line 362, in get_device_properties
    raise AssertionError("Invalid device id")
AssertionError: Invalid device id


----------------------------------------------------------------------------
Training XGBoost with config/adult.yml in env gbdt

Namespace(config='config/adult.yml', model_name='XGBoost', dataset='Adult', objective='binary', use_gpu=False, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=5, shuffle=True, seed=221, scale=True, target_encode=True, one_hot_encode=False, batch_size=128, val_batch_size=256, early_stopping_rounds=20, epochs=300, logging_period=100, num_features=14, num_classes=1, cat_idx=[1, 3, 5, 6, 7, 8, 9, 13], cat_dims=[9, 16, 7, 15, 6, 5, 2, 42])    
Train model with given hyperparameters
Loading dataset Adult...
Dataset loaded!
(32561, 14)
Scaling the data...
[0]     eval-auc:0.86786
[100]   eval-auc:0.91725
[200]   eval-auc:0.92200
[299]   eval-auc:0.92303
{'Log Loss - mean': 0.28761527874017456, 'Log Loss - std': 0.0, 'AUC - mean': 0.9230288730273647, 'AUC - std': 0.0, 'Accuracy - mean': 0.8638108398587441, 'Accuracy - std': 0.0, 'F1 score - mean': 0.8638108398587441, 'F1 score - std': 0.0}
[0]     eval-auc:0.88021
[100]   eval-auc:0.92405
[200]   eval-auc:0.92958
[299]   eval-auc:0.93103
{'Log Loss - mean': 0.2803434343163548, 'Log Loss - std': 0.007271844423819812, 'AUC - mean': 0.9270278358206139, 'AUC - std': 0.003998962793249183, 'Accuracy - mean': 0.8704035771775294, 'Accuracy - std': 0.006592737318785236, 'F1 score - mean': 0.8704035771775294, 'F1 score - std': 0.006592737318785236}
[0]     eval-auc:0.88692
[100]   eval-auc:0.92955
[200]   eval-auc:0.93501
[299]   eval-auc:0.93614
{'Log Loss - mean': 0.2749989181732772, 'Log Loss - std': 0.009611495885486657, 'AUC - mean': 0.930066362531258, 'AUC - std': 0.0053968902858148055, 'Accuracy - mean': 0.8737272824099173, 'Accuracy - std': 0.007146338678881211, 'F1 score - mean': 0.8737272824099173, 'F1 score - std': 0.007146338678881211}
[0]     eval-auc:0.87333
[100]   eval-auc:0.92065
[200]   eval-auc:0.92585
[280]   eval-auc:0.92667
{'Log Loss - mean': 0.2762761836498431, 'Log Loss - std': 0.00861277313996058, 'AUC - mean': 0.9292162637758377, 'AUC - std': 0.004900287976205398, 'Accuracy - mean': 0.8738535084904846, 'Accuracy - std': 0.0061927713065477764, 'F1 score - mean': 0.8738535084904846, 'F1 score - std': 0.0061927713065477764}
[0]     eval-auc:0.87765
[100]   eval-auc:0.92116
[200]   eval-auc:0.92544
[299]   eval-auc:0.92634
{'Log Loss - mean': 0.27732255546081036, 'Log Loss - std': 0.007982697839420513, 'AUC - mean': 0.9286420420250622, 'AUC - std': 0.0045309138455778605, 'Accuracy - mean': 0.8734071311167121, 'Accuracy - std': 0.00561046737834074, 'F1 score - mean': 0.8734071311167121, 'F1 score - std': 0.00561046737834074}
{'Log Loss - mean': 0.27732255546081036, 'Log Loss - std': 0.007982697839420513, 'AUC - mean': 0.9286420420250622, 'AUC - std': 0.0045309138455778605, 'Accuracy - mean': 0.8734071311167121, 'Accuracy - std': 0.00561046737834074, 'F1 score - mean': 0.8734071311167121, 'F1 score - std': 0.00561046737834074}
(57.39637646000001, 0.025482460000006313)


----------------------------------------------------------------------------
Training MLP with config/adult.yml in env torch

Namespace(config='config/adult.yml', model_name='MLP', dataset='Adult', objective='binary', use_gpu=False, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=5, shuffle=True, seed=221, scale=True, target_encode=True, one_hot_encode=False, batch_size=128, val_batch_size=256, early_stopping_rounds=20, epochs=300, logging_period=100, num_features=14, num_classes=1, cat_idx=[1, 3, 5, 6, 7, 8, 9, 13], cat_dims=[9, 16, 7, 15, 6, 5, 2, 42])        
Train model with given hyperparameters
Loading dataset Adult...
Dataset loaded!
(32561, 14)
Scaling the data...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 131, in main_once
    model = model_name(parameters, args)
  File "/opt/notebooks/models/mlp.py", line 22, in __init__
    self.to_device()
  File "/opt/notebooks/models/basemodel_torch.py", line 22, in to_device
    self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/torch/nn/parallel/data_parallel.py", line 142, in __init__
    _check_balance(self.device_ids)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/torch/nn/parallel/data_parallel.py", line 23, in _check_balance
    dev_props = _get_devices_properties(device_ids)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/torch/_utils.py", line 491, in _get_devices_properties
    return [_get_device_attr(lambda m: m.get_device_properties(i)) for i in device_ids]
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/torch/_utils.py", line 491, in <listcomp>
    return [_get_device_attr(lambda m: m.get_device_properties(i)) for i in device_ids]
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/torch/_utils.py", line 474, in _get_device_attr
    return get_member(torch.cuda)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/torch/_utils.py", line 491, in <lambda>
    return [_get_device_attr(lambda m: m.get_device_properties(i)) for i in device_ids]
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/torch/cuda/__init__.py", line 362, in get_device_properties
    raise AssertionError("Invalid device id")
AssertionError: Invalid device id


----------------------------------------------------------------------------
Training DecisionTree with config/adult.yml in env sklearn

Namespace(config='config/adult.yml', model_name='DecisionTree', dataset='Adult', objective='binary', use_gpu=False, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=5, shuffle=True, seed=221, scale=True, target_encode=True, one_hot_encode=False, batch_size=128, val_batch_size=256, early_stopping_rounds=20, epochs=300, logging_period=100, num_features=14, num_classes=1, cat_idx=[1, 3, 5, 6, 7, 8, 9, 13], cat_dims=[9, 16, 7, 15, 6, 5, 2, 42])
Train model with given hyperparameters
Loading dataset Adult...
Dataset loaded!
(32561, 14)
Scaling the data...
{'Log Loss - mean': 0.40767437306381726, 'Log Loss - std': 0.0, 'AUC - mean': 0.8968444668238383, 'AUC - std': 0.0, 'Accuracy - mean': 0.8460003070781514, 'Accuracy - std': 0.0, 'F1 score - mean': 0.8460003070781514, 'F1 score - std': 0.0}
{'Log Loss - mean': 0.4094290160823269, 'Log Loss - std': 0.0017546430185096384, 'AUC - mean': 0.8988172313686262, 'AUC - std': 0.0019727645447878417, 'Accuracy - mean': 0.852591676880599, 'Accuracy - std': 0.006591369802447622, 'F1 score - mean': 0.852591676880599, 'F1 score - std': 0.006591369802447622}
{'Log Loss - mean': 0.4108952747405974, 'Log Loss - std': 0.0025203856161659286, 'AUC - mean': 0.9006349645345914, 'AUC - std': 0.003033618470530981, 'Accuracy - mean': 0.8544304872897687, 'Accuracy - std': 0.005977169175857456, 'F1 score - mean': 0.8544304872897687, 'F1 score - std': 0.005977169175857456}
{'Log Loss - mean': 0.4121238435916238, 'Log Loss - std': 0.003048344141623281, 'AUC - mean': 0.9001276832719065, 'AUC - std': 0.0027702226637895803, 'Accuracy - mean': 0.8540062192756803, 'Accuracy - std': 0.005228281128722024, 'F1 score - mean': 0.8540062192756803, 'F1 score - std': 0.005228281128722024}
{'Log Loss - mean': 0.41361392378405465, 'Log Loss - std': 0.004039217439568465, 'AUC - mean': 0.9003425740991975, 'AUC - std': 0.002514760258194414, 'Accuracy - mean': 0.8542430589586278, 'Accuracy - std': 0.004700245843616618, 'F1 score - mean': 0.8542430589586278, 'F1 score - std': 0.004700245843616618}
{'Log Loss - mean': 0.41361392378405465, 'Log Loss - std': 0.004039217439568465, 'AUC - mean': 0.9003425740991975, 'AUC - std': 0.002514760258194414, 'Accuracy - mean': 0.8542430589586278, 'Accuracy - std': 0.004700245843616618, 'F1 score - mean': 0.8542430589586278, 'F1 score - std': 0.004700245843616618}
(0.08608251999999998, 0.005573240000000102)


----------------------------------------------------------------------------
Training DeepGBM with config/adult.yml in env torch

Namespace(config='config/adult.yml', model_name='DeepGBM', dataset='Adult', objective='binary', use_gpu=False, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=5, shuffle=True, seed=221, scale=True, target_encode=True, one_hot_encode=False, batch_size=128, val_batch_size=256, early_stopping_rounds=20, epochs=300, logging_period=100, num_features=14, num_classes=1, cat_idx=[1, 3, 5, 6, 7, 8, 9, 13], cat_dims=[9, 16, 7, 15, 6, 5, 2, 42])    
Train model with given hyperparameters
Loading dataset Adult...
Dataset loaded!
(32561, 14)
Scaling the data...
{'task': 'binary', 'batch_size': 128, 'test_batch_size': 256, 'l2_reg': 1e-06, 'l2_reg_opt': 0.0005, 'lr': 0.001, 'epochs': 300, 'early_stopping': 3, 'loss_de': 10, 'loss_dr': 0.7, 'device': device(type='cpu'), 'bins': 32, 'rate': 0.9, 'threshold': 10, 'maxleaf': 64, 'num_slices': 5, 'n_feature': 128, 'n_clusters': 10, 'tree_lr': 0.1, 'n_trees': 200, 'embsize': 20, 'emb_lr': 0.001, 'emb_epochs': 3, 'tree_layers': [100, 100, 100, 50, 20], 'embedding_size': 20, 'cate_layers': [16, 16], 'online_epochs': 1, 'online_bz': 4096, 'num_splits': 5, 'early-stopping': 20}
{'task': 'binary', 'batch_size': 128, 'test_batch_size': 256, 'l2_reg': 1e-06, 'l2_reg_opt': 0.0005, 'lr': 0.001, 'epochs': 300, 'early_stopping': 3, 'loss_de': 10, 'loss_dr': 0.7, 'device': device(type='cpu'), 'bins': 32, 'rate': 0.9, 'threshold': 10, 'maxleaf': 64, 'num_slices': 5, 'n_feature': 128, 'n_clusters': 10, 'tree_lr': 0.1, 'n_trees': 200, 'embsize': 20, 'emb_lr': 0.001, 'emb_epochs': 3, 'tree_layers': [100, 100, 100, 50, 20], 'embedding_size': 20, 'cate_layers': [16, 16], 'online_epochs': 1, 'online_bz': 4096, 'num_splits': 5, 'early-stopping': 20}
Preprocess data for CatNN...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 133, in main_once
    sc, time = cross_validation(model, X, y, args)
  File "/opt/notebooks/train.py", line 41, in cross_validation
    loss_history, val_loss_history = curr_model.fit(X_train, y_train, X_test, y_test)  # X_val, y_val)
  File "/opt/notebooks/models/deepgbm.py", line 48, in fit
    train_x_cat, feature_sizes = self.ce.fit_transform(X.copy())
  File "/opt/notebooks/models/deepgbm_lib/preprocess/preprocessing_cat.py", line 35, in fit_transform
    out, bins = pd.qcut(X[:, idx], config.config['bins'], labels=False, retbins=True, duplicates='drop')
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/core/reshape/tile.py", line 376, in qcut
    x_np = x_np[~np.isnan(x_np)]
TypeError: ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''     
(base) root@4ab780bd064d:/opt/notebooks# exit