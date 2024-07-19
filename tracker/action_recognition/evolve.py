import os
from functools import partial

import optuna
import joblib
import json

from tracker.action_recognition.evaluate import AREvaluator
from tracker.action_recognition.generate_behaviors import generate_behaviors
from tracker.finetune.evolve import generate_unique_tag


def ar_optuna_fitness_fn(trial, config, objective, param_names):
    ###############################
    # OVERRIDE CONFIG WITH SOLUTION
    ###############################

    include_projection = config["evolve"]["projection"]
    include_buffer = config["evolve"]["buffer"]

    # Update the config with the solution
    ar_config = config["action_recognition"]
    if include_projection:
        ar_config["speed_projection"][0] = trial.suggest_float("speed_projection_x", 0.5, 2.0)
        ar_config["speed_projection"][1] = trial.suggest_float("speed_projection_y", 0.5, 2.0)

    if include_buffer:
        ar_config["speed_buffer_len"] = trial.suggest_int("speed_buffer_len", 1, 15)
        ar_config["frame_stride"] = trial.suggest_int("frame_stride", 1, 60)

    if objective == "Macro" or objective == "Micro":
        ar_config["gather"]["distance_threshold"] = trial.suggest_float("g_distance_threshold", 0.1, 1.5)
        ar_config["gather"]["area_threshold"] = trial.suggest_float("g_area_threshold", 0.1, 1.0)
        ar_config["gather"]["speed_threshold"] = trial.suggest_float("g_speed_threshold", 0.0001, 0.1)
        ar_config["stand_still"]["speed_threshold"] = trial.suggest_float("ss_speed_threshold", 0.0001, 0.1)
        ar_config["fast_approach"]["speed_threshold"] = trial.suggest_float("fa_speed_threshold", 0.0001, 0.1)
        ar_config["suddenly_run"]["speed_threshold"] = trial.suggest_float("sr_speed_threshold", 0.0001, 0.1)

    elif objective == "G":
        ar_config["gather"]["distance_threshold"] = trial.suggest_float("g_distance_threshold", 0.1, 1.5)
        ar_config["gather"]["area_threshold"] = trial.suggest_float("g_area_threshold", 0.1, 1.0)
        ar_config["gather"]["speed_threshold"] = trial.suggest_float("g_speed_threshold", 0.0001, 0.1)

        ar_config["stand_still"]["enabled"] = False
        ar_config["fast_approach"]["enabled"] = False
        ar_config["suddenly_run"]["enabled"] = False

    elif objective == "SS":
        ar_config["stand_still"]["speed_threshold"] = trial.suggest_float("ss_speed_threshold", 0.0001, 0.1)

        ar_config["gather"]["enabled"] = False
        ar_config["fast_approach"]["enabled"] = False
        ar_config["suddenly_run"]["enabled"] = False

    elif objective == "FA":
        ar_config["fast_approach"]["speed_threshold"] = trial.suggest_float("fa_speed_threshold", 0.0001, 0.1)

        ar_config["gather"]["enabled"] = False
        ar_config["stand_still"]["enabled"] = False
        ar_config["suddenly_run"]["enabled"] = False

    elif objective == "SR":
        ar_config["suddenly_run"]["speed_threshold"] = trial.suggest_float("sr_speed_threshold", 0.0001, 0.1)

        ar_config["gather"]["enabled"] = False
        ar_config["stand_still"]["enabled"] = False
        ar_config["fast_approach"]["enabled"] = False

    elif objective == "projection":
        ar_config["speed_projection"][0] = trial.suggest_float("speed_projection_x", 0.5, 2.0)
        ar_config["speed_projection"][1] = trial.suggest_float("speed_projection_y", 0.5, 2.0)
        objective = "Macro"
    else:
        raise ValueError(f"Objective {objective} not recognized")

    ###############################
    # PERFORM INFERENCE AND SAVE CONFIG
    ###############################
    # Perform Inference on the dataset
    experiment_id = generate_unique_tag()
    processor = generate_behaviors(
        config=config,
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
    eval_config["data_dir"] = config["source_gt_dir"] + "/"
    eval_config["action_recognition"]["smoothing_window"] = trial.suggest_int("smoothing_window", 0, 60, step=10) if "smoothing_window" in param_names else 0
    eval_config["action_recognition"]["final_pad"] = trial.suggest_int("final_pad", 0, 60, step=10) if "final_pad" in param_names else 0
    eval_config["action_recognition"]["initial_pad"] = trial.suggest_int("initial_pad", 0, 200, step=10) if "initial_pad" in param_names else 0
    eval_config["action_recognition"]["save_results"] = False
    eval_config["action_recognition"]["print_results"] = False
    eval_config["action_recognition"]["discriminate_groups"] = False

    if objective in ["SS", "FA", "SR", "G"]:
        eval_config["action_recognition"]["active_behaviors"] = [objective]
    else:
        eval_config["action_recognition"]["active_behaviors"] = ["SS", "FA", "SR", "G"]

    evaluator = AREvaluator(eval_config)
    metrics_df = evaluator.evaluate()

    print(f"Trial {trial.number}:")
    print(metrics_df.to_string(float_format="{:.4f}".format, index=False, col_space=10))

    return metrics_df[metrics_df['Class']==objective]['F'].values[0]


def print_and_save(study, trial):
    studies_path = "./outputs/studies"
    if not os.path.exists(studies_path):
        os.makedirs(studies_path)

    joblib.dump(study, f"./outputs/studies/{study.study_name}.pkl")


if __name__ == "__main__":
    resume = False
    if resume:
        study = joblib.load(f"./outputs/studies/optuna/no-name-5c697fc9-b4cb-4485-8b73-3df5eacda8fc_study.pkl")
    else:
        study = optuna.create_study(direction="maximize")

    # Load config from default params
    with open("./cfg/evolve.json", "r") as f:
        config = json.load(f)

    objective = config["evolve"]["objective"]
    include_projection = config["evolve"]["projection"]

    initial_params = {}
    params_to_optimize = []
    if objective in ["SS", "Macro", "Micro"]:
        params_to_optimize.append("ss_speed_threshold")
        initial_params["ss_speed_threshold"] = config["action_recognition"]["stand_still"]["speed_threshold"]

    if objective in ["FA", "Macro", "Micro"]:
        params_to_optimize.append("fa_speed_threshold")
        initial_params["fa_speed_threshold"] = config["action_recognition"]["fast_approach"]["speed_threshold"]

    if objective in ["SR", "Macro", "Micro"]:
        params_to_optimize.append("sr_speed_threshold")
        initial_params["sr_speed_threshold"] = config["action_recognition"]["suddenly_run"]["speed_threshold"]

    if objective in ["G", "Macro", "Micro"]:
        params_to_optimize.append("g_distance_threshold")
        params_to_optimize.append("g_area_threshold")
        params_to_optimize.append("g_speed_threshold")
        initial_params["g_distance_threshold"] = config["action_recognition"]["gather"]["distance_threshold"]
        initial_params["g_area_threshold"] = config["action_recognition"]["gather"]["area_threshold"]
        initial_params["g_speed_threshold"] = config["action_recognition"]["gather"]["speed_threshold"]

    if include_projection or objective == "projection":
        params_to_optimize = ["speed_projection_x", "speed_projection_y"] + params_to_optimize
        initial_params["speed_projection_x"] = config["action_recognition"]["speed_projection"][0]
        initial_params["speed_projection_y"] = config["action_recognition"]["speed_projection"][1]

    if config["evolve"]["post_process"]:
        params_to_optimize += ["initial_pad", "final_pad", "smoothing_window"]
        initial_params["initial_pad"] = 0
        initial_params["final_pad"] = 0
        initial_params["smoothing_window"] = 0

    if config["evolve"]["buffer"]:
        params_to_optimize += ["speed_buffer_len", "frame_stride"]
        initial_params["speed_buffer_len"] = config["action_recognition"]["speed_buffer_len"]
        initial_params["frame_stride"] = config["action_recognition"]["frame_stride"]

    study.enqueue_trial(initial_params)

    # We could add a continuous save function to save the study every 10 trials and print the best trial
    study.optimize(
        func=partial(ar_optuna_fitness_fn, config=config, objective=objective, param_names=params_to_optimize),
        n_trials=config["evolve"]["n_trials"],
        show_progress_bar=True,
        callbacks=[print_and_save]
    )

    print("\nStudy Statistics: ")
    print("Best Trial:      ", study.best_trial.number)
    print("Best Value:      ", study.best_value)
    print("Best Parameters: ")
    for key, value in study.best_params.items():
        print(f"\t{key}: {value}")
