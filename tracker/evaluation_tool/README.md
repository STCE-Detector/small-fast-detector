# 🛠️ Evaluation Tool

This tool is used to evaluate the tracker's performance. 📈

## 🛠️ Before You Begin

Make sure you have the following data:

1. Data for sequence inference:
   - 📁 **evaluation_tool/data_mot**: Directory containing the MOT dataset (MOT20-01, MOT20-02, MOT20-03, MOT20-05)
   - 📁 **evaluation_tool/TrackEval/data**: Directory containing the ground truth to evaluate the tracker

You can download it [here](https://drive.google.com/drive/folders/1fx_3wimHxZl4lxRDApQJRrPbkaD4abG4?usp=drive_link).
## 🚀 Usage

```bash
python3 scripts/run_mot_challenge.py --TRACKERS_TO_EVAL <path_to_tracker_output> --iou_thresh <iou_threshold>
```
