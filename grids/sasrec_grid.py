import optuna

def generate_config(trial: optuna.Trial) -> dict:
    config = dict(
        # trial params
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256]), #[64, 128] #, 256, 512]  
        learning_rate = trial.suggest_categorical('learning_rate', [0.0001, 0.001, 0.005]), #[0.00001, 0.0001, 0.001]  
        hidden_units = trial.suggest_categorical('hidden_units', [32, 64, 128, 256, 512]), # [32, 64] #, 128] #, 256, 512, 768]  
        num_blocks = trial.suggest_categorical('num_blocks', [1, 2, 3]), #, 2, 3]  
        num_heads = trial.suggest_categorical('num_heads', [1]),
        dropout_rate = trial.suggest_categorical('dropout_rate', [0.2, 0.4, 0.6]), # [0.2, 0.4, 0.6]  
        l2_emb = trial.suggest_categorical('l2_emb', [0.0]),
        # fixed params:
        maxlen = 200,
        classifier = False,
        neg_sampling = False,
        seed = 0,
        sampler_seed = 789,
        device = None,
        max_epochs = 200,
        step_replacement = 'max_epochs' # during tests, max value will be replaced with an optimal one
    )
    return config

