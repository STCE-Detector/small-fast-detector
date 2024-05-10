import optuna
import joblib
import json

from tracker.finetune.byte_mot20_fitness import optuna_fitness_fn

def print_and_save(study, trial):
    #print("Trial Number: ", trial.number)
    #print("Study Best Value: ", study.best_value)
    #print("Study Best Params: ", study.best_params)
    #print("Study Best Trial: ", study.best_trial.number)

    joblib.dump(study, f"./outputs/studies/optuna/{study.study_name}_study.pkl")


if __name__ == "__main__":

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
    study.optimize(func=optuna_fitness_fn, n_trials=200, show_progress_bar=True, callbacks=[print_and_save])

    print("\nStudy Statistics: ")
    print("Best Trial:      ", study.best_trial.number)
    print("Best Value:      ", study.best_value)
    print("Best Parameters: ")
    for key, value in study.best_params.items():
        print(f"\t{key}: {value}")

