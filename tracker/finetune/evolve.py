import os
import random
import string
import time

import optuna
import joblib
import json

from functools import partial

from tracker.evaluation.TrackEval_evaluate import trackeval
from tracker.evaluation.generate_tracks import generate_tracks


def generate_unique_tag():
    timestamp = int(time.time())  # Current Unix timestamp
    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))  # Random 6-character alphanumeric string
    tag = f"exp_{timestamp}_{random_suffix}"
    return tag


def optuna_fitness_fn(trial, config):
    """
    Fitness function for the Optuna optimization library. This function evaluates the fitness of a solution by running
    the tracker with the specified parameters and evaluating the results using the TrackEval evaluation script.
    Args:
        trial: The Optuna trial object
    Returns:
        The fitness value of the solution
    """

    # Update the config with the solution
    tracker_config = config["tracker_args"]
    tracker_config["track_high_thresh"] = trial.suggest_float("track_high_thresh", 0.3, 0.9, step=0.01)
    #tracker_config["track_low_thresh"] = trial.suggest_float("track_low_thresh", 0.1, tracker_config["track_high_thresh"], step=0.01)
    #tracker_config["new_track_thresh"] = trial.suggest_float("new_track_thresh", tracker_config["track_low_thresh"], 0.9, step=0.01)
    tracker_config["track_low_thresh"] = trial.suggest_float("track_low_thresh", 0.1, 0.4, step=0.01)
    tracker_config["new_track_thresh"] = trial.suggest_float("new_track_thresh", 0.1, 0.9, step=0.01)
    tracker_config["first_match_thresh"] = trial.suggest_float("first_match_thresh", 0.5, 1.0, step=0.01)
    tracker_config["second_match_thresh"] = trial.suggest_float("second_match_thresh", 0.4, 1.0, step=0.01)
    tracker_config["new_match_thresh"] = trial.suggest_float("new_match_thresh", 0.5, 1.0, step=0.01)
    tracker_config["first_buffer"] = trial.suggest_float("first_buffer", 0.0, 0.5, step=0.01)
    tracker_config["second_buffer"] = trial.suggest_float("second_buffer", 0.0, 0.5, step=0.01)
    tracker_config["new_buffer"] = trial.suggest_float("new_buffer", 0.0, 0.5, step=0.01)
    tracker_config["first_fuse"] = trial.suggest_int("first_fuse", 0, 1)
    tracker_config["second_fuse"] = trial.suggest_int("second_fuse", 0, 1)
    tracker_config["new_fuse"] = trial.suggest_int("new_fuse", 0, 1)
    tracker_config["first_iou"] = trial.suggest_int("first_iou", 0, 5)
    tracker_config["second_iou"] = trial.suggest_int("second_iou", 0, 5)
    tracker_config["new_iou"] = trial.suggest_int("new_iou", 0, 5)
    tracker_config["cw_thresh"] = trial.suggest_float("cw_thresh", 0.0, 1.0, step=0.01)
    tracker_config["nk_flag"] = trial.suggest_int("nk_flag", 0, 1)
    tracker_config["nk_alpha"] = trial.suggest_int("nk_alpha", 0, 150, step=10)
    tracker_config["track_buffer"] = trial.suggest_int("track_buffer", 0, 500, step=50)

    # Perform Inference on the dataset
    experiment_id = generate_unique_tag()
    processor = generate_tracks(
        config=config,
        experiment_id=experiment_id
    )

    # Save the tracker config
    trackers_folder = os.path.abspath("./outputs/tracks/" + processor.dataset)
    trackers_to_eval = processor.experiment_name
    json_path = trackers_folder + "/" + trackers_to_eval + "/config.json"
    with open(json_path, "w") as f:
        json.dump(tracker_config, f, indent=4)

    # Run evaluation
    combined_metrics = trackeval(
        evaluator_folder=os.path.abspath("./../evaluation/TrackEval"),
        dataset_folder=config["source_gt_dir"],
        trackers_folder=trackers_folder,
        trackers_to_eval=trackers_to_eval,
        metrics=["HOTA"],  # uncomment this line to only evaluate HOTA, single-object fitness
    )
    return combined_metrics["HOTA"]
    #return list(combined_metrics.values())


def print_and_save(study, trial, name):
    studies_path = "./outputs/studies"
    if not os.path.exists(studies_path):
        os.makedirs(studies_path)

    joblib.dump(study, f"./outputs/studies/{name}.pkl")


if __name__ == "__main__":

    resume = False
    if resume:
        study = joblib.load(f"./outputs/studies/optuna/no-name-5c697fc9-b4cb-4485-8b73-3df5eacda8fc_study.pkl")
    else:
        study = optuna.create_study(direction="maximize")

    # Load config from default params
    with open("./cfg/evolve.json", "r") as f:
        config = json.load(f)

    params_to_optimize = ["track_high_thresh", "track_low_thresh", "new_track_thresh", "first_match_thresh",
                          "second_match_thresh", "new_match_thresh", "first_buffer", "second_buffer", "new_buffer",
                          "first_fuse", "second_fuse", "new_fuse", "first_iou", "second_iou", "new_iou", "cw_thresh",
                          "nk_flag", "nk_alpha", "track_buffer"]

    initial_params = {key: config["tracker_args"][key] for key in params_to_optimize}
    # Enqueue trial for good starting point
    study.enqueue_trial(initial_params)

    # We could add a continuous save function to save the study every 10 trials and print the best trial
    study.optimize(
        func=partial(optuna_fitness_fn, config=config),
        n_trials=config["n_trials"],
        show_progress_bar=True,
        callbacks=[partial(print_and_save, name=config["name"])]
    )

    print("\nStudy Statistics: ")
    print("Best Trial:      ", study.best_trial.number)
    print("Best Value:      ", study.best_value)
    print("Best Parameters: ")
    for key, value in study.best_params.items():
        print(f"\t{key}: {value}")

