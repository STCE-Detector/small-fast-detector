import optuna
import joblib
import json

from functools import partial

from tracker.evaluation.generate_tracks import generate_tracks
from tracker.finetune.byte_mot_fitness import generate_unique_tag

def ar_optuna_fitness_fn(trial, names):
    # Read common config
    with open("./../cfg/ByteTrack.json", "r") as f:
        default_config = json.load(f)

    # Update the config with the solution
    ar_config = default_config["action_recognition"]
    ar_config["speed_projection"][0] = trial.suggest_float("speed_projection_x", 0.0, 1.0)
    ar_config["speed_projection"][1] = trial.suggest_float("speed_projection_y", 0.0, 1.0)
    ar_config["gather"]["distance_threshold"] = trial.suggest_float("g_distance_threshold", 0.0, 1.0)
    ar_config["gather"]["area_threshold"] = trial.suggest_float("g_area_threshold", 0.0, 1.0)
    ar_config["stand_still"]["speed_threshold"] = trial.suggest_float("ss_speed_threshold", 0.0, 1.0)
    ar_config["fast_approach"]["speed_threshold"] = trial.suggest_float("fa_speed_threshold", 0.0, 1.0)
    ar_config["suddenly_run"]["speed_threshold"] = trial.suggest_float("sr_speed_threshold", 0.0, 1.0)

    # Perform Inference on the dataset
    experiment_id = generate_unique_tag()
    # Use default track_recognize but no display, only save the results in the desired folder

    processor = generate_tracks(
        config=default_config,
        experiment_id=experiment_id,
        ar=True
    )

    # Generate tracks

    # Evaluate tracks



def print_and_save(study, trial):
    #print("Trial Number: ", trial.number)
    #print("Study Best Value: ", study.best_value)
    #print("Study Best Params: ", study.best_params)
    #print("Study Best Trial: ", study.best_trial.number)

    joblib.dump(study, f"./outputs/studies/optuna/{study.study_name}_study.pkl")


if __name__ == "__main__":

    resume = False
    if resume:
        study = joblib.load(f"./outputs/studies/optuna/no-name-5c697fc9-b4cb-4485-8b73-3df5eacda8fc_study.pkl")
    else:
        study = optuna.create_study(direction="maximize")

    # Load config from default params
    with open("./cfg/evolve.json", "r") as f:
        config = json.load(f)

    params_to_optimize = ["speed_projection_x", "speed_projection_y",  "g_distance_threshold", "g_area_threshold",
                          "ss_speed_threshold", "fa_speed_threshold", "sr_speed_threshold"]

    initial_params = {key: config["tracker_args"][key] for key in params_to_optimize}
    # Enqueue trial for good starting point
    study.enqueue_trial(initial_params)

    # We could add a continuous save function to save the study every 10 trials and print the best trial
    objective = partial(ar_optuna_fitness_fn, names=params_to_optimize)
    study.optimize(func=objective, n_trials=400, show_progress_bar=True, callbacks=[print_and_save])

    print("\nStudy Statistics: ")
    print("Best Trial:      ", study.best_trial.number)
    print("Best Value:      ", study.best_value)
    print("Best Parameters: ")
    for key, value in study.best_params.items():
        print(f"\t{key}: {value}")