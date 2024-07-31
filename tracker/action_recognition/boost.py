import os
from functools import partial

import optuna
import joblib
import json

from tracker.action_recognition.evaluate import AREvaluator
from tracker.action_recognition.generate_behaviors import generate_behaviors
from tracker.finetune.evolve import generate_unique_tag


def ar_optuna_fitness_fn(trial, config, trackers_folder, trackers_to_eval):
    """
    Fitness function for the Optuna optimization of the action recognition parameters
    :param trial: Optuna trial object
    :param config: Configuration dictionary
    :param trackers_folder: Path to the folder containing the trackers
    :param trackers_to_eval: Name of the tracker to evaluate
    :return: The value of the objective metric
    """
    objective = config["evolve"]["objective"][0]
    metric = config["evolve"]["objective"][1]

    # Read evaluation config
    with open("./cfg/eval.json", "r") as f:
        eval_config = json.load(f)

    # Override the config to include the experiment_id in the pred_dir and disable printing the confusion matrix
    eval_config["name"] = None
    eval_config["pred_dir"] = trackers_folder + "/" + trackers_to_eval + "/"
    eval_config["data_dir"] = config["source_gt_dir"] + "/"
    eval_config["action_recognition"]["smoothing_window"] = trial.suggest_int("smoothing_window", 0, 30)
    eval_config["action_recognition"]["final_pad"] = trial.suggest_int("final_pad", 0, 30)
    eval_config["action_recognition"]["initial_pad"] = trial.suggest_int("initial_pad", 0, 30)
    eval_config["action_recognition"]["save_results"] = False
    eval_config["action_recognition"]["print_results"] = False
    eval_config["action_recognition"]["discriminate_groups"] = False
    eval_config["action_recognition"]["active_behaviors"] = ["SS", "SR", "G", "FA", "OB"]

    evaluator = AREvaluator(eval_config)
    metrics_df = evaluator.evaluate()

    print(f"Trial {trial.number}:")
    print(metrics_df.to_string(float_format="{:.4f}".format, index=False, col_space=10))

    return metrics_df[metrics_df['Class']==objective][metric].values[0]


def print_and_save(study, trial, name):
    """
    Print the best trial and save the study
    :param study: Optuna study object
    :param trial: Optuna trial object
    :param name: Name of the study
    """
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
    with open("./cfg/boost.json", "r") as f:
        config = json.load(f)

    # Generate the predictions, that will be static
    experiment_id = generate_unique_tag()
    processor = generate_behaviors(
        config=config,
        experiment_id=experiment_id,
        print_bar=False
    )
    # Save the config file
    trackers_folder = os.path.abspath("./outputs/tracks/" + processor.dataset)
    trackers_to_eval = processor.experiment_name

    # Define the parameters to optimize and initialize the study
    params_to_optimize = ["initial_pad", "final_pad", "smoothing_window"]
    initial_params = {"initial_pad": 0, "final_pad": 0, "smoothing_window": 0}

    study.enqueue_trial(initial_params)

    # We could add a continuous save function to save the study every 10 trials and print the best trial
    study.optimize(
        func=partial(ar_optuna_fitness_fn,
                     config=config,
                     trackers_folder=trackers_folder,
                     trackers_to_eval=trackers_to_eval),
        n_trials=config["evolve"]["n_trials"],
        show_progress_bar=True,
        callbacks=[partial(print_and_save, name=config["name"])]
    )

    print("\nStudy Statistics: ")
    print("Best Trial:      ", study.best_trial.number)
    print("Best Value:      ", study.best_value)
    print("Best Parameters: ")
    for key, value in study.best_params.items():
        print(f"\t{key}: {value}")
