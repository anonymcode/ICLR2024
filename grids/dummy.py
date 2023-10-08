import optuna


def generate_config(trial: optuna.Trial) -> dict:
    config = dict(
        # trial params
        seed = trial.suggest_int('seed', 0, 2**31-1),
        # fixed params:
        max_epochs = 100,
        step_replacement = 'max_epochs' # during tests, max value will be replaced with an optimal one
    )
    return config