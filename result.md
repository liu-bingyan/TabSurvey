Scaling the data...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 133, in main_once
    sc, time = cross_validation(model, X, y, args)
  File "/opt/notebooks/train.py", line 41, in cross_validation     
    loss_history, val_loss_history = curr_model.fit(X_train, y_train, X_test, y_test)  # X_val, y_val)
  File "/opt/notebooks/models/modeltree.py", line 28, in fit       
    X = np.array(X, dtype=np.float)
  File "/opt/conda/envs/gbdt/lib/python3.9/site-packages/numpy/__init__.py", line 324, in __getattr__
    raise AttributeError(__former_attrs__[attr])
AttributeError: module 'numpy' has no attribute 'float'.
`np.float` was a deprecated alias for the builtin `float`. To avoid this error in existing code, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations


----------------------------------------------------------------------------
Training NAM with config/california_housing.yml in env torch       

Namespace(config='config/california_housing.yml', model_name='NAM', dataset='CaliforniaHousing', objective='regression', use_gpu=False, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='minimize', num_splits=5, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=128, val_batch_size=256, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=8, num_classes=1, cat_idx=None, cat_dims=None)
Train model with given hyperparameters
Loading dataset CaliforniaHousing...
Dataset loaded!
(20640, 8)
Scaling the data...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 128, in main_once
    model_name = str2model(args.model_name)
  File "/opt/notebooks/models/__init__.py", line 81, in str2model  
    from models.neural_additive_models import NAM
  File "/opt/notebooks/models/neural_additive_models.py", line 4, in <module>
    from nam.config import defaults
ModuleNotFoundError: No module named 'nam'


----------------------------------------------------------------------------
Training STG with config/california_housing.yml in env torch       

Namespace(config='config/california_housing.yml', model_name='STG', dataset='CaliforniaHousing', objective='regression', use_gpu=False, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='minimize', num_splits=5, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=128, val_batch_size=256, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=8, num_classes=1, cat_idx=None, cat_dims=None)
Train model with given hyperparameters
Loading dataset CaliforniaHousing...
Dataset loaded!
(20640, 8)
Scaling the data...
Epoch: 1: loss=2.430160 valid_loss=1.053670
Epoch: 2: loss=1.062783 valid_loss=0.805892
Epoch: 3: loss=0.891754 valid_loss=0.706239
{'MSE - mean': 0.7123070245208912, 'MSE - std': 0.0, 'R2 - mean': 0.47481415814374006, 'R2 - std': 0.0}
Epoch: 1: loss=1.779427 valid_loss=0.799974
Epoch: 2: loss=1.014488 valid_loss=0.719403
Epoch: 3: loss=0.961216 valid_loss=0.692520
{'MSE - mean': 0.7052202656975961, 'MSE - std': 0.0070867588232950696, 'R2 - mean': 0.4665662172197781, 'R2 - std': 0.008247940923961972}
Epoch: 1: loss=1.875993 valid_loss=0.828825
Epoch: 2: loss=0.967747 valid_loss=0.720818
Epoch: 3: loss=0.904988 valid_loss=0.710878
{'MSE - mean': 0.7091816266694435, 'MSE - std': 0.008053955254712879, 'R2 - mean': 0.462717504392464, 'R2 - std': 0.008658956856595068}
Epoch: 1: loss=2.045305 valid_loss=0.920902
Epoch: 2: loss=0.971580 valid_loss=0.791534
Epoch: 3: loss=0.937290 valid_loss=0.753858
{'MSE - mean': 0.7226680685217587, 'MSE - std': 0.024378309784341527, 'R2 - mean': 0.4599906198148259, 'R2 - std': 0.008862327509326333}
Epoch: 1: loss=2.105684 valid_loss=0.870165
Epoch: 2: loss=0.993114 valid_loss=0.746585
Epoch: 3: loss=0.973359 valid_loss=0.703198
{'MSE - mean': 0.719499602501054, 'MSE - std': 0.022706789690541854, 'R2 - mean': 0.45947548482603073, 'R2 - std': 0.007993380719869029}
{'MSE - mean': 0.719499602501054, 'MSE - std': 0.022706789690541854, 'R2 - mean': 0.45947548482603073, 'R2 - std': 0.007993380719869029}
(5.1088420800000005, 0.08699455999999976)


----------------------------------------------------------------------------
Training RandomForest with config/california_housing.yml in env sklearn

Namespace(config='config/california_housing.yml', model_name='RandomForest', dataset='CaliforniaHousing', objective='regression', use_gpu=False, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='minimize', num_splits=5, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=128, val_batch_size=256, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=8, num_classes=1, cat_idx=None, cat_dims=None)
Train model with given hyperparameters
Loading dataset CaliforniaHousing...
Dataset loaded!
(20640, 8)
Scaling the data...
{'MSE - mean': 0.29134512315923045, 'MSE - std': 0.0, 'R2 - mean': 0.7851904747394379, 'R2 - std': 0.0}
{'MSE - mean': 0.29464852187946394, 'MSE - std': 0.003303398720233497, 'R2 - mean': 0.7770048034898025, 'R2 - std': 0.008185671249635484}
{'MSE - mean': 0.29916983131706637, 'MSE - std': 0.006939700291150819, 'R2 - mean': 0.7732588253226157, 'R2 - std': 0.00852847274699016}
{'MSE - mean': 0.30044455768113987, 'MSE - std': 0.0064026839690899335, 'R2 - mean': 0.7753015093745432, 'R2 - std': 0.00818955493022134}
{'MSE - mean': 0.2946067637715871, 'MSE - std': 0.01300441619328126, 'R2 - mean': 0.7785960833221246, 'R2 - std': 0.00985250820595539}
{'MSE - mean': 0.2946067637715871, 'MSE - std': 0.01300441619328126, 'R2 - mean': 0.7785960833221246, 'R2 - std': 0.00985250820595539}
(0.7235133599999999, 0.010348920000000029)


----------------------------------------------------------------------------
Training LightGBM with config/higgs.yml in env gbdt

Namespace(config='config/higgs.yml', model_name='LightGBM', dataset='HIGGS', objective='binary', use_gpu=True, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=3, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=1024, val_batch_size=1024, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=28, num_classes=1, cat_idx=[27], cat_dims=None)
Train model with given hyperparameters
Loading dataset HIGGS...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 126, in main_once
    X, y = load_data(args)
  File "/opt/notebooks/utils/load_data.py", line 78, in load_data  
    df = pd.read_csv(path, header=None)
  File "/opt/conda/envs/gbdt/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 948, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/opt/conda/envs/gbdt/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 611, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/opt/conda/envs/gbdt/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1448, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/opt/conda/envs/gbdt/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1705, in _make_engine
    self.handles = get_handle(
  File "/opt/conda/envs/gbdt/lib/python3.9/site-packages/pandas/io/common.py", line 755, in get_handle
    handle = gzip.GzipFile(  # type: ignore[assignment]
  File "/opt/conda/envs/gbdt/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/opt/notebooks/data/HIGGS.csv.gz'


----------------------------------------------------------------------------
Training KNN with config/higgs.yml in env sklearn

Namespace(config='config/higgs.yml', model_name='KNN', dataset='HIGGS', objective='binary', use_gpu=True, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=3, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=1024, val_batch_size=1024, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=28, num_classes=1, cat_idx=[27], cat_dims=None)
Train model with given hyperparameters
Loading dataset HIGGS...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 126, in main_once
    X, y = load_data(args)
  File "/opt/notebooks/utils/load_data.py", line 78, in load_data  
    df = pd.read_csv(path, header=None)
  File "/opt/conda/envs/sklearn/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 948, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/opt/conda/envs/sklearn/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 611, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/opt/conda/envs/sklearn/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1448, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/opt/conda/envs/sklearn/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1705, in _make_engine
    self.handles = get_handle(
  File "/opt/conda/envs/sklearn/lib/python3.9/site-packages/pandas/io/common.py", line 755, in get_handle
    handle = gzip.GzipFile(  # type: ignore[assignment]
  File "/opt/conda/envs/sklearn/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/opt/notebooks/data/HIGGS.csv.gz'


----------------------------------------------------------------------------
Training VIME with config/higgs.yml in env torch

Namespace(config='config/higgs.yml', model_name='VIME', dataset='HIGGS', objective='binary', use_gpu=True, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=3, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=1024, val_batch_size=1024, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=28, num_classes=1, cat_idx=[27], cat_dims=None)
Train model with given hyperparameters
Loading dataset HIGGS...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 126, in main_once
    X, y = load_data(args)
  File "/opt/notebooks/utils/load_data.py", line 78, in load_data  
    df = pd.read_csv(path, header=None)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 948, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 611, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1448, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1705, in _make_engine
    self.handles = get_handle(
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/common.py", line 755, in get_handle
    handle = gzip.GzipFile(  # type: ignore[assignment]
  File "/opt/conda/envs/torch/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/opt/notebooks/data/HIGGS.csv.gz'


----------------------------------------------------------------------------
Training LinearModel with config/higgs.yml in env sklearn

Namespace(config='config/higgs.yml', model_name='LinearModel', dataset='HIGGS', objective='binary', use_gpu=True, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=3, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=1024, val_batch_size=1024, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=28, num_classes=1, cat_idx=[27], cat_dims=None)      
Train model with given hyperparameters
Loading dataset HIGGS...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 126, in main_once
    X, y = load_data(args)
  File "/opt/notebooks/utils/load_data.py", line 78, in load_data  
    df = pd.read_csv(path, header=None)
  File "/opt/conda/envs/sklearn/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 948, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/opt/conda/envs/sklearn/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 611, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/opt/conda/envs/sklearn/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1448, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/opt/conda/envs/sklearn/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1705, in _make_engine
    self.handles = get_handle(
  File "/opt/conda/envs/sklearn/lib/python3.9/site-packages/pandas/io/common.py", line 755, in get_handle
    handle = gzip.GzipFile(  # type: ignore[assignment]
  File "/opt/conda/envs/sklearn/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/opt/notebooks/data/HIGGS.csv.gz'


----------------------------------------------------------------------------
Training TabTransformer with config/higgs.yml in env torch

Namespace(config='config/higgs.yml', model_name='TabTransformer', dataset='HIGGS', objective='binary', use_gpu=True, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=3, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=1024, val_batch_size=1024, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=28, num_classes=1, cat_idx=[27], cat_dims=None)   
Train model with given hyperparameters
Loading dataset HIGGS...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 126, in main_once
    X, y = load_data(args)
  File "/opt/notebooks/utils/load_data.py", line 78, in load_data  
    df = pd.read_csv(path, header=None)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 948, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 611, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1448, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1705, in _make_engine
    self.handles = get_handle(
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/common.py", line 755, in get_handle
    handle = gzip.GzipFile(  # type: ignore[assignment]
  File "/opt/conda/envs/torch/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/opt/notebooks/data/HIGGS.csv.gz'


----------------------------------------------------------------------------
Training NODE with config/higgs.yml in env torch

Namespace(config='config/higgs.yml', model_name='NODE', dataset='HIGGS', objective='binary', use_gpu=True, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=3, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=1024, val_batch_size=1024, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=28, num_classes=1, cat_idx=[27], cat_dims=None)
Train model with given hyperparameters
Loading dataset HIGGS...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 126, in main_once
    X, y = load_data(args)
  File "/opt/notebooks/utils/load_data.py", line 78, in load_data  
    df = pd.read_csv(path, header=None)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 948, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 611, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1448, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1705, in _make_engine
    self.handles = get_handle(
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/common.py", line 755, in get_handle
    handle = gzip.GzipFile(  # type: ignore[assignment]
  File "/opt/conda/envs/torch/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/opt/notebooks/data/HIGGS.csv.gz'


----------------------------------------------------------------------------
Training SAINT with config/higgs.yml in env torch

Namespace(config='config/higgs.yml', model_name='SAINT', dataset='HIGGS', objective='binary', use_gpu=True, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=3, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=1024, val_batch_size=1024, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=28, num_classes=1, cat_idx=[27], cat_dims=None)
Train model with given hyperparameters
Loading dataset HIGGS...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 126, in main_once
    X, y = load_data(args)
  File "/opt/notebooks/utils/load_data.py", line 78, in load_data  
    df = pd.read_csv(path, header=None)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 948, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 611, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1448, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1705, in _make_engine
    self.handles = get_handle(
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/common.py", line 755, in get_handle
    handle = gzip.GzipFile(  # type: ignore[assignment]
  File "/opt/conda/envs/torch/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/opt/notebooks/data/HIGGS.csv.gz'


----------------------------------------------------------------------------
Training TabNet with config/higgs.yml in env torch

Namespace(config='config/higgs.yml', model_name='TabNet', dataset='HIGGS', objective='binary', use_gpu=True, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=3, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=1024, val_batch_size=1024, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=28, num_classes=1, cat_idx=[27], cat_dims=None)
Train model with given hyperparameters
Loading dataset HIGGS...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 126, in main_once
    X, y = load_data(args)
  File "/opt/notebooks/utils/load_data.py", line 78, in load_data  
    df = pd.read_csv(path, header=None)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 948, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 611, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1448, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1705, in _make_engine
    self.handles = get_handle(
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/common.py", line 755, in get_handle
    handle = gzip.GzipFile(  # type: ignore[assignment]
  File "/opt/conda/envs/torch/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/opt/notebooks/data/HIGGS.csv.gz'


----------------------------------------------------------------------------
Training DeepFM with config/higgs.yml in env torch

Namespace(config='config/higgs.yml', model_name='DeepFM', dataset='HIGGS', objective='binary', use_gpu=True, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=3, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=1024, val_batch_size=1024, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=28, num_classes=1, cat_idx=[27], cat_dims=None)
Train model with given hyperparameters
Loading dataset HIGGS...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 126, in main_once
    X, y = load_data(args)
  File "/opt/notebooks/utils/load_data.py", line 78, in load_data  
    df = pd.read_csv(path, header=None)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 948, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 611, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1448, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1705, in _make_engine
    self.handles = get_handle(
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/common.py", line 755, in get_handle
    handle = gzip.GzipFile(  # type: ignore[assignment]
  File "/opt/conda/envs/torch/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/opt/notebooks/data/HIGGS.csv.gz'


----------------------------------------------------------------------------
Training XGBoost with config/higgs.yml in env gbdt

Namespace(config='config/higgs.yml', model_name='XGBoost', dataset='HIGGS', objective='binary', use_gpu=True, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=3, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=1024, val_batch_size=1024, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=28, num_classes=1, cat_idx=[27], cat_dims=None)
Train model with given hyperparameters
Loading dataset HIGGS...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 126, in main_once
    X, y = load_data(args)
  File "/opt/notebooks/utils/load_data.py", line 78, in load_data  
    df = pd.read_csv(path, header=None)
  File "/opt/conda/envs/gbdt/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 948, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/opt/conda/envs/gbdt/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 611, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/opt/conda/envs/gbdt/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1448, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/opt/conda/envs/gbdt/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1705, in _make_engine
    self.handles = get_handle(
  File "/opt/conda/envs/gbdt/lib/python3.9/site-packages/pandas/io/common.py", line 755, in get_handle
    handle = gzip.GzipFile(  # type: ignore[assignment]
  File "/opt/conda/envs/gbdt/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/opt/notebooks/data/HIGGS.csv.gz'


----------------------------------------------------------------------------
Training RLN with config/higgs.yml in env tensorflow

Namespace(config='config/higgs.yml', model_name='RLN', dataset='HIGGS', objective='binary', use_gpu=True, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=3, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=1024, val_batch_size=1024, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=28, num_classes=1, cat_idx=[27], cat_dims=None)
Train model with given hyperparameters
Loading dataset HIGGS...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 126, in main_once
    X, y = load_data(args)
  File "/opt/notebooks/utils/load_data.py", line 78, in load_data  
    df = pd.read_csv(path, header=None)
  File "/opt/conda/envs/tensorflow/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 948, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/opt/conda/envs/tensorflow/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 611, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/opt/conda/envs/tensorflow/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1448, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/opt/conda/envs/tensorflow/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1705, in _make_engine
    self.handles = get_handle(
  File "/opt/conda/envs/tensorflow/lib/python3.9/site-packages/pandas/io/common.py", line 755, in get_handle
    handle = gzip.GzipFile(  # type: ignore[assignment]
  File "/opt/conda/envs/tensorflow/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/opt/notebooks/data/HIGGS.csv.gz'


----------------------------------------------------------------------------
Training MLP with config/higgs.yml in env torch

Namespace(config='config/higgs.yml', model_name='MLP', dataset='HIGGS', objective='binary', use_gpu=True, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=3, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=1024, val_batch_size=1024, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=28, num_classes=1, cat_idx=[27], cat_dims=None)
Train model with given hyperparameters
Loading dataset HIGGS...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 126, in main_once
    X, y = load_data(args)
  File "/opt/notebooks/utils/load_data.py", line 78, in load_data  
    df = pd.read_csv(path, header=None)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 948, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 611, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1448, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1705, in _make_engine
    self.handles = get_handle(
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/common.py", line 755, in get_handle
    handle = gzip.GzipFile(  # type: ignore[assignment]
  File "/opt/conda/envs/torch/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/opt/notebooks/data/HIGGS.csv.gz'


----------------------------------------------------------------------------
Training DecisionTree with config/higgs.yml in env sklearn

Namespace(config='config/higgs.yml', model_name='DecisionTree', dataset='HIGGS', objective='binary', use_gpu=True, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=3, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=1024, val_batch_size=1024, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=28, num_classes=1, cat_idx=[27], cat_dims=None)     
Train model with given hyperparameters
Loading dataset HIGGS...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 126, in main_once
    X, y = load_data(args)
  File "/opt/notebooks/utils/load_data.py", line 78, in load_data  
    df = pd.read_csv(path, header=None)
  File "/opt/conda/envs/sklearn/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 948, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/opt/conda/envs/sklearn/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 611, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/opt/conda/envs/sklearn/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1448, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/opt/conda/envs/sklearn/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1705, in _make_engine
    self.handles = get_handle(
  File "/opt/conda/envs/sklearn/lib/python3.9/site-packages/pandas/io/common.py", line 755, in get_handle
    handle = gzip.GzipFile(  # type: ignore[assignment]
  File "/opt/conda/envs/sklearn/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/opt/notebooks/data/HIGGS.csv.gz'


----------------------------------------------------------------------------
Training DNFNet with config/higgs.yml in env tensorflow

Namespace(config='config/higgs.yml', model_name='DNFNet', dataset='HIGGS', objective='binary', use_gpu=True, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=3, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=1024, val_batch_size=1024, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=28, num_classes=1, cat_idx=[27], cat_dims=None)
Train model with given hyperparameters
Loading dataset HIGGS...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 126, in main_once
    X, y = load_data(args)
  File "/opt/notebooks/utils/load_data.py", line 78, in load_data  
    df = pd.read_csv(path, header=None)
  File "/opt/conda/envs/tensorflow/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 948, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/opt/conda/envs/tensorflow/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 611, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/opt/conda/envs/tensorflow/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1448, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/opt/conda/envs/tensorflow/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1705, in _make_engine
    self.handles = get_handle(
  File "/opt/conda/envs/tensorflow/lib/python3.9/site-packages/pandas/io/common.py", line 755, in get_handle
    handle = gzip.GzipFile(  # type: ignore[assignment]
  File "/opt/conda/envs/tensorflow/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/opt/notebooks/data/HIGGS.csv.gz'


----------------------------------------------------------------------------
Training DeepGBM with config/higgs.yml in env torch

Namespace(config='config/higgs.yml', model_name='DeepGBM', dataset='HIGGS', objective='binary', use_gpu=True, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=3, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=1024, val_batch_size=1024, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=28, num_classes=1, cat_idx=[27], cat_dims=None)
Train model with given hyperparameters
Loading dataset HIGGS...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 126, in main_once
    X, y = load_data(args)
  File "/opt/notebooks/utils/load_data.py", line 78, in load_data  
    df = pd.read_csv(path, header=None)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 948, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 611, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1448, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1705, in _make_engine
    self.handles = get_handle(
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/common.py", line 755, in get_handle
    handle = gzip.GzipFile(  # type: ignore[assignment]
  File "/opt/conda/envs/torch/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/opt/notebooks/data/HIGGS.csv.gz'


----------------------------------------------------------------------------
Training CatBoost with config/higgs.yml in env gbdt

Namespace(config='config/higgs.yml', model_name='CatBoost', dataset='HIGGS', objective='binary', use_gpu=True, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=3, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=1024, val_batch_size=1024, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=28, num_classes=1, cat_idx=[27], cat_dims=None)
Train model with given hyperparameters
Loading dataset HIGGS...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 126, in main_once
    X, y = load_data(args)
  File "/opt/notebooks/utils/load_data.py", line 78, in load_data  
    df = pd.read_csv(path, header=None)
  File "/opt/conda/envs/gbdt/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 948, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/opt/conda/envs/gbdt/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 611, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/opt/conda/envs/gbdt/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1448, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/opt/conda/envs/gbdt/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1705, in _make_engine
    self.handles = get_handle(
  File "/opt/conda/envs/gbdt/lib/python3.9/site-packages/pandas/io/common.py", line 755, in get_handle
    handle = gzip.GzipFile(  # type: ignore[assignment]
  File "/opt/conda/envs/gbdt/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/opt/notebooks/data/HIGGS.csv.gz'


----------------------------------------------------------------------------
Training DANet with config/higgs.yml in env torch

Namespace(config='config/higgs.yml', model_name='DANet', dataset='HIGGS', objective='binary', use_gpu=True, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=3, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=1024, val_batch_size=1024, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=28, num_classes=1, cat_idx=[27], cat_dims=None)
Train model with given hyperparameters
Loading dataset HIGGS...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 126, in main_once
    X, y = load_data(args)
  File "/opt/notebooks/utils/load_data.py", line 78, in load_data  
    df = pd.read_csv(path, header=None)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 948, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 611, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1448, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1705, in _make_engine
    self.handles = get_handle(
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/common.py", line 755, in get_handle
    handle = gzip.GzipFile(  # type: ignore[assignment]
  File "/opt/conda/envs/torch/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/opt/notebooks/data/HIGGS.csv.gz'


----------------------------------------------------------------------------
Training ModelTree with config/higgs.yml in env gbdt

Namespace(config='config/higgs.yml', model_name='ModelTree', dataset='HIGGS', objective='binary', use_gpu=True, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=3, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=1024, val_batch_size=1024, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=28, num_classes=1, cat_idx=[27], cat_dims=None)        
Train model with given hyperparameters
Loading dataset HIGGS...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 126, in main_once
    X, y = load_data(args)
  File "/opt/notebooks/utils/load_data.py", line 78, in load_data  
    df = pd.read_csv(path, header=None)
  File "/opt/conda/envs/gbdt/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 948, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/opt/conda/envs/gbdt/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 611, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/opt/conda/envs/gbdt/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1448, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/opt/conda/envs/gbdt/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1705, in _make_engine
    self.handles = get_handle(
  File "/opt/conda/envs/gbdt/lib/python3.9/site-packages/pandas/io/common.py", line 755, in get_handle
    handle = gzip.GzipFile(  # type: ignore[assignment]
  File "/opt/conda/envs/gbdt/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/opt/notebooks/data/HIGGS.csv.gz'


----------------------------------------------------------------------------
Training NAM with config/higgs.yml in env torch

Namespace(config='config/higgs.yml', model_name='NAM', dataset='HIGGS', objective='binary', use_gpu=True, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=3, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=1024, val_batch_size=1024, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=28, num_classes=1, cat_idx=[27], cat_dims=None)
Train model with given hyperparameters
Loading dataset HIGGS...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 126, in main_once
    X, y = load_data(args)
  File "/opt/notebooks/utils/load_data.py", line 78, in load_data  
    df = pd.read_csv(path, header=None)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 948, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 611, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1448, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1705, in _make_engine
    self.handles = get_handle(
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/common.py", line 755, in get_handle
    handle = gzip.GzipFile(  # type: ignore[assignment]
  File "/opt/conda/envs/torch/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/opt/notebooks/data/HIGGS.csv.gz'


----------------------------------------------------------------------------
Training STG with config/higgs.yml in env torch

Namespace(config='config/higgs.yml', model_name='STG', dataset='HIGGS', objective='binary', use_gpu=True, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=3, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=1024, val_batch_size=1024, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=28, num_classes=1, cat_idx=[27], cat_dims=None)
Train model with given hyperparameters
Loading dataset HIGGS...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 126, in main_once
    X, y = load_data(args)
  File "/opt/notebooks/utils/load_data.py", line 78, in load_data  
    df = pd.read_csv(path, header=None)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 948, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 611, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1448, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1705, in _make_engine
    self.handles = get_handle(
  File "/opt/conda/envs/torch/lib/python3.9/site-packages/pandas/io/common.py", line 755, in get_handle
    handle = gzip.GzipFile(  # type: ignore[assignment]
  File "/opt/conda/envs/torch/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/opt/notebooks/data/HIGGS.csv.gz'


----------------------------------------------------------------------------
Training RandomForest with config/higgs.yml in env sklearn

Namespace(config='config/higgs.yml', model_name='RandomForest', dataset='HIGGS', objective='binary', use_gpu=True, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=3, shuffle=True, seed=221, scale=True, target_encode=False, one_hot_encode=False, batch_size=1024, val_batch_size=1024, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=28, num_classes=1, cat_idx=[27], cat_dims=None)     
Train model with given hyperparameters
Loading dataset HIGGS...
Traceback (most recent call last):
  File "/opt/notebooks/train.py", line 149, in <module>
    main_once(arguments)
  File "/opt/notebooks/train.py", line 126, in main_once
    X, y = load_data(args)
  File "/opt/notebooks/utils/load_data.py", line 78, in load_data  
    df = pd.read_csv(path, header=None)
  File "/opt/conda/envs/sklearn/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 948, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/opt/conda/envs/sklearn/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 611, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/opt/conda/envs/sklearn/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1448, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/opt/conda/envs/sklearn/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1705, in _make_engine
    self.handles = get_handle(
  File "/opt/conda/envs/sklearn/lib/python3.9/site-packages/pandas/io/common.py", line 755, in get_handle
    handle = gzip.GzipFile(  # type: ignore[assignment]
  File "/opt/conda/envs/sklearn/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/opt/notebooks/data/HIGGS.csv.gz'