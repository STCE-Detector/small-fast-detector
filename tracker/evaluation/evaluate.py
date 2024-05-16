import os
import re
import sys
import subprocess


def trackeval(
        evaluator_folder: str,
        dataset_folder: str,
        trackers_folder: str,
        trackers_to_eval: str,
        metrics=["HOTA", "CLEAR", "Identity"],
        print_output=False
):
    """
    Run the TrackEval evaluation script.
    Args:
        evaluator_folder: Path to the TrackEval folder.
        dataset_folder: Path to the dataset folder.
        trackers_folder: Path to the trackers folder.
        trackers_to_eval: Name of the tracker to evaluate.
        metrics: List of metrics to evaluate.
        print_output: Whether to print the output of the evaluation script.
    Returns:
        Dictionary with the results of the evaluation.
    """

    seq_info = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]

    args = [
        sys.executable, evaluator_folder + '/scripts/run_mot_challenge.py',
        "--GT_FOLDER", dataset_folder,
        "--SEQ_INFO", *seq_info,
        "--TRACKERS_FOLDER", trackers_folder,
        "--TRACKERS_TO_EVAL", trackers_to_eval,
        "--METRICS", *metrics,
        "--USE_PARALLEL", "True",
        "--NUM_PARALLEL_CORES", str(4),
        "--SKIP_SPLIT_FOL", "True",
        "--SPLIT_TO_EVAL", "train"
    ]

    # Execute the command
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Get the output
    out, err = p.communicate()

    # Output the result
    if print_output:
        print("STDOUT:\n", out)
        if err:
            print("STDERR:\n", err)

    # Parse the output
    combined_results = out.split('COMBINED')[2:-1]
    # robust way of getting first ints/float in string
    combined_results = [float(re.findall("[-+]?(?:\d*\.*\d+)", f)[0]) for f in combined_results]
    # pack everything in dict
    combined_results = {key: value for key, value in zip(['HOTA', 'MOTA', 'IDF1'], combined_results)}

    return combined_results


if __name__ == "__main__":
    result = trackeval(
        evaluator_folder="/Users/inaki-eab/Desktop/small-fast-detector/tracker/evaluation/TrackEval",
        dataset_folder="/Users/inaki-eab/Desktop/small-fast-detector/tracker/evaluation/TrackEval/data/gt/mot_challenge/MOTHupba-train",
        trackers_folder="/Users/inaki-eab/Desktop/small-fast-detector/tracker/evaluation/outputs/tracks/MOTHupba-train",
        trackers_to_eval="ByteTrack_YOLOv8s-P2-640_2",
        print_output=True
    )
    print(result)

