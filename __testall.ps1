$N_TRIALS = 5
$EPOCHS = 400

$SKLEARN_ENV = "sklearn"
$GBDT_ENV = "gbdt"
$TORCH_ENV = "torch"
$KERAS_ENV = "tensorflow"

$MODELS = @{
    "LinearModel" = $SKLEARN_ENV
    #"KNN" = $SKLEARN_ENV
    #"SVM" = $SKLEARN_ENV
    #"DecisionTree" = $SKLEARN_ENV
    #"RandomForest" = $SKLEARN_ENV
    #"XGBoost" = $GBDT_ENV
    #"CatBoost" = $GBDT_ENV
    #"LightGBM" = $GBDT_ENV
    #"MLP" = $TORCH_ENV
    #"TabNet" = $TORCH_ENV
    #"VIME" = $TORCH_ENV
    #"TabTransformer" = $TORCH_ENV
    #"ModelTree" = $GBDT_ENV
    #"NODE" = $TORCH_ENV
    #"DeepGBM" = $TORCH_ENV
    #"RLN" = $KERAS_ENV
    #"DNFNet" = $KERAS_ENV
    #"STG" = $TORCH_ENV
    #"NAM" = $TORCH_ENV
    #"DeepFM" = $TORCH_ENV
    #"SAINT" = $TORCH_ENV
    #"DANet" = $TORCH_ENV
}

$CONFIGS = @(
    "config/adult.yml"
    #"config/covertype.yml"
    #"config/california_housing.yml"
    #"config/higgs.yml"
)

#$condaScriptPath = (Get-Command conda).Source

foreach ($config in $CONFIGS) {
    foreach ($model in $MODELS.Keys) {
        Write-Host ("`n`n----------------------------------------------------------------------------")
        Write-Host ("Training $model with $config in env $($MODELS[$model])`n")

        #& $condaScriptPath activate $($MODELS[$model])

        python train.py --config "$config" --model_name "$model" --n_trials $N_TRIALS --epochs $EPOCHS

        #& $condaScriptPath deactivate
    }
}
