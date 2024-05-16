import json
import os.path
import time
import random
import string

from tracker.evaluation.generate_tracks import generate_tracks
from tracker.evaluation.evaluate import trackeval


def generate_unique_tag():
    timestamp = int(time.time())  # Current Unix timestamp
    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))  # Random 6-character alphanumeric string
    tag = f"exp_{timestamp}_{random_suffix}"
    return tag


def fitness_fn(ga_instance, solution, solution_idx):
    """
    Fitness function for the genetic algorithm. This function evaluates the fitness of a solution by running the tracker
    with the specified parameters and evaluating the results using the TrackEval evaluation script.
    Args:
        ga_instance: The genetic algorithm instance
        solution: The solution to evaluate
        solution_idx: The index of the solution in the population
    Returns:
        The fitness value of the solution
    """

    # Read common config
    with open("./cfg/evolve.json", "r") as f:
        config = json.load(f)

    # Update the config with the solution
    tracker_config = config["tracker_args"]
    for gene_value, gene_key in zip(solution, tracker_config):
        tracker_config[gene_key] = gene_value

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
        json.dump(tracker_config, f)

    # Run evaluation
    combined_metrics = trackeval(
        evaluator_folder=os.path.abspath("./../evaluation/TrackEval"),
        dataset_folder=config["source_gt_dir"],
        trackers_folder=trackers_folder,
        trackers_to_eval=trackers_to_eval,
        metrics=["HOTA"],   # uncomment this line to only evaluate HOTA, single-object fitness
    )
    print(combined_metrics)
    return combined_metrics["HOTA"] #list(combined_metrics.values())


def optuna_fitness_fn(trial):
    """
    Fitness function for the Optuna optimization library. This function evaluates the fitness of a solution by running
    the tracker with the specified parameters and evaluating the results using the TrackEval evaluation script.
    Args:
        trial: The Optuna trial object
    Returns:
        The fitness value of the solution
    """

    # Read common config
    with open("./cfg/evolve.json", "r") as f:
        config = json.load(f)

    # Update the config with the solution
    tracker_config = config["tracker_args"]
    tracker_config["track_high_thresh"] = trial.suggest_float("track_high_thresh", 0.2, 0.9, step=0.01)
    tracker_config["track_low_thresh"] = trial.suggest_float("track_low_thresh", 0.1, 0.4, step=0.01)
    tracker_config["new_track_thresh"] = trial.suggest_float("new_track_thresh", 0.2, 0.6, step=0.01)
    tracker_config["first_match_thresh"] = trial.suggest_float("first_match_thresh", 0.2, 1.0, step=0.01)
    tracker_config["second_match_thresh"] = trial.suggest_float("second_match_thresh", 0.2, 1.0, step=0.01)
    tracker_config["new_match_thresh"] = trial.suggest_float("new_match_thresh", 0.2, 1.0, step=0.01)
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
        json.dump(tracker_config, f)

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


if __name__ == "__main__":
    solution = [0.6, 0.3]
    h = fitness_fn(None, solution, 0)
    print(h)

