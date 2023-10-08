
# ========= DATA-RELEATED =========
# data location
data_dir = './data'

# data description
timeid = 'timestamp'
userid = 'userid'

# data preprocessing
use_cached_pcore = True # download ready-to-use amazon files instead of preprocessing

# data splits
time_offset_q = 0.95
max_test_interactions = 100_000


# ========= MODEL-RELEATED =========
validation_interval = 20 # frequency of validation for iterative models; 1 means validate on each iteration


# ========= STUDY-RELEATED =========
study_direction = 'maximize'
max_attempts_multiplier = 3
drop_duplicated_trials = False


# ========= OPTUNA-RELEATED =========
grid_steps_limit = 60
disable_experimental_warnings = True