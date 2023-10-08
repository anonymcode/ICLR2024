import optuna

def generate_config(trial: optuna.Trial) -> dict:
    config = dict(
        # trial params
        batch_size = trial.suggest_categorical('batch_size', [128]), 
        learning_rate = trial.suggest_categorical('learning_rate', [0.001]),
        hidden_units = trial.suggest_categorical('hidden_units', [512]),
        num_blocks = trial.suggest_categorical('num_blocks', [3]), #, 2, 3]  
        num_heads = trial.suggest_categorical('num_heads', [1]),
        dropout_rate = trial.suggest_categorical('dropout_rate', [0.1, 0.2, 0.3]),
        l2_emb = trial.suggest_categorical('l2_emb', [0.0]),
        # fixed params:
        maxlen = 200,
        seed = 0,
        sampler_seed = 789,
        device = None,
        max_epochs = 400,
        step_replacement = 'max_epochs' # during tests, max value will be replaced with an optimal one
    )
    return config
