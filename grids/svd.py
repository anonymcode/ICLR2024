import optuna


def generate_config(trial: optuna.Trial) -> dict:
    config = dict(
        # trial params
        rank = trial.suggest_categorical('rank', [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]),
        scaling = trial.suggest_categorical('scaling', [0.0, 0.2, 0.4, 0.6, 1.0]),
        # fixed params:
        randomized = True
    )
    if config['randomized']:
        config['rnd_svd_seed'] = 0
    return config