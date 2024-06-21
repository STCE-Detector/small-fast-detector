import os
import optuna
import joblib
import json

from tracker.ar_tools.evaluate import AREvaluator
from tracker.ar_tools.generate_behaviors import generate_behaviors
from tracker.finetune.evolve import generate_unique_tag


def ar_optuna_fitness_fn(trial):
    ###############################
    # OVERRIDE CONFIG WITH SOLUTION
    ###############################
    # Read common config
    with open("./cfg/evolve.json", "r") as f:
        default_config = json.load(f)

    # Update the config with the solution
    # TODO: set appropriate values for the parameters
    ar_config = default_config["action_recognition"]
    ar_config["speed_projection"][0] = trial.suggest_float("speed_projection_x", 0.5, 2.0)
    ar_config["speed_projection"][1] = trial.suggest_float("speed_projection_y", 0.5, 2.0)
    ar_config["gather"]["distance_threshold"] = trial.suggest_float("g_distance_threshold", 0.1, 1.5)
    ar_config["gather"]["area_threshold"] = trial.suggest_float("g_area_threshold", 0.1, 1.0)
    ar_config["stand_still"]["speed_threshold"] = trial.suggest_float("ss_speed_threshold", 0.0001, 0.1)
    ar_config["fast_approach"]["speed_threshold"] = trial.suggest_float("fa_speed_threshold", 0.0001, 0.1)
    ar_config["suddenly_run"]["speed_threshold"] = trial.suggest_float("sr_speed_threshold", 0.0001, 0.1)

    ###############################
    # PERFORM INFERENCE AND SAVE CONFIG
    ###############################
    # Perform Inference on the dataset
    experiment_id = generate_unique_tag()
    processor = generate_behaviors(
        config=default_config,
        experiment_id=experiment_id,
        print_bar=False
    )

    # Save the config file
    trackers_folder = os.path.abspath("./outputs/tracks/" + processor.dataset)
    trackers_to_eval = processor.experiment_name
    json_path = trackers_folder + "/" + trackers_to_eval + "/config.json"
    with open(json_path, "w") as f:
        json.dump(ar_config, f)

    ###############################
    # EVALUATE THE PERFORMANCE
    ###############################
    # Read evaluation config
    with open("./cfg/eval.json", "r") as f:
        eval_config = json.load(f)

    # Override the config to include the experiment_id in the pred_dir and disable printing the confusion matrix
    eval_config["pred_dir"] = trackers_folder + "/" + trackers_to_eval + "/"
    eval_config["action_recognition"]["smoothing_window"] = 0   # TODO: Set to 0 for now
    eval_config["action_recognition"]["save_results"] = False
    eval_config["action_recognition"]["print_results"] = False

    evaluator = AREvaluator(eval_config)
    metrics_df = evaluator.evaluate()

    return metrics_df[metrics_df['Class'] == 'Macro']['F2'].values[0]


def print_and_save(study, trial):
    output_root = "./outputs/studies/"
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)
    joblib.dump(study, output_root + f"{study.study_name}_study.pkl")


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

    # Enqueue trial for good starting point
    initial_params = {
        "speed_projection_x": config["action_recognition"]["speed_projection"][0],
        "speed_projection_y": config["action_recognition"]["speed_projection"][1],
        "g_distance_threshold": config["action_recognition"]["gather"]["distance_threshold"],
        "g_area_threshold": config["action_recognition"]["gather"]["area_threshold"],
        "ss_speed_threshold": config["action_recognition"]["stand_still"]["speed_threshold"],
        "fa_speed_threshold": config["action_recognition"]["fast_approach"]["speed_threshold"],
        "sr_speed_threshold": config["action_recognition"]["suddenly_run"]["speed_threshold"]
    }
    study.enqueue_trial(initial_params)

    # We could add a continuous save function to save the study every 10 trials and print the best trial
    study.optimize(func=ar_optuna_fitness_fn, n_trials=100, show_progress_bar=True, callbacks=[print_and_save])

    print("\nStudy Statistics: ")
    print("Best Trial:      ", study.best_trial.number)
    print("Best Value:      ", study.best_value)
    print("Best Parameters: ")
    for key, value in study.best_params.items():
        print(f"\t{key}: {value}")