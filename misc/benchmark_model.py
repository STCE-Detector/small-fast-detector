import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info

"""
These functions collectively provide a toolkit for evaluating different aspects of a model's performance, from its computational complexity (parameters, FLOPs) to its real-world applicability (inference time, FPS, throughput). By benchmarking models on these metrics, developers can make informed decisions about which models are most suitable for their applications, balancing the trade-offs between accuracy, speed, and computational resources.
"""

import matplotlib.pyplot as plt


def plot_results(df, filename='benchmark_results.png'):
    # Set up the plotting environment
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    models = df['model']

    # Plot for Parameters Count and GFLOPs
    ax1 = axs[0]
    ax1.bar(models, df['parameters (count)'].str.replace(',', '').astype(float), alpha=0.6, label='Parameters (count)')
    ax1.set_ylabel('Parameters (count)')
    ax1.set_title('Model Complexity: Parameters and GFLOPs')
    ax1.tick_params(axis='x', rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(models, df['GFLOPs'].str.replace(',', '').astype(float), color='r', marker='o', label='GFLOPs')
    ax2.set_ylabel('GFLOPs')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    # Plot for Latency and FPS
    ax3 = axs[1]
    ax3.plot(models, df['latency (ms)'].str.replace(',', '').astype(float), color='g', marker='x', linestyle='-',
             label='Latency (ms)')
    ax3.set_ylabel('Latency (ms)')
    ax3.set_title('Model Performance: Latency and FPS')
    ax3.tick_params(axis='x', rotation=45)

    ax4 = ax3.twinx()
    ax4.plot(models, df['FPS (frames/s)'].str.replace(',', '').astype(float), color='b', marker='o', linestyle='--',
             label='FPS (frames/s)')
    ax4.set_ylabel('FPS (frames/s)')
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax4.legend(lines3 + lines4, labels3 + labels4, loc=0)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def check_file_exists(path, arch):
    """
    Purpose: To verify that a specified configuration file for the model exists at a given path. This is a utility function to ensure necessary files are present before attempting to load or benchmark a model.
    How It Works: It constructs the full path to the expected file and checks if the file exists there, raising an error if not. This helps prevent runtime errors due to missing files.
    """
    full_path = os.path.join(path, arch) + '.yaml'
    print(full_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"El archivo {full_path} no existe.")
    else:
        print(f"El archivo {full_path} existe.")
    return full_path


def inference_yolo_hard(cfg, yolo):
    """
    Purpose: The core function that orchestrates the entire benchmarking process for a list of model architectures.
    How It Works: For each architecture, it performs a series of measurements (like parameter count, FLOPs, inference time, FPS, and latency) using the functions described above. It aggregates these metrics into a comprehensive report (in the form of a DataFrame), allowing for an in-depth comparison of model performances.
    """
    pre = []
    infer = []
    post = []

    for i in range(cfg.num_runs):
        # Generate a random image
        image = torch.rand(1, 3, cfg.imgsz, cfg.imgsz).to(cfg.device)

        # Run inference
        results = yolo.predict(
            source=image,
            imgsz=cfg.imgsz,
            device=cfg.device,
            verbose=False,
        )

        if i >= cfg.num_warmup_runs:
            # Collect inference speed
            pre.append(results[0].speed['preprocess'])
            infer.append(results[0].speed['inference'])
            post.append(results[0].speed['postprocess'])

    # Calculate the mean inference speeds
    t_pre = np.mean(pre)
    t_infer = np.mean(infer)
    t_post = np.mean(post)

    return t_pre, t_infer, t_post


def perform_benchmark(cfg, archs, path='../ultralytics/cfg/models/v8/'):
    results = {
        "Model": [],
        "Parameters": [],  # Sin unidad específica, conteo de parámetros
        "GFLOPs": [],  # Giga Floating Point Operations, sin unidad de tiempo porque es un conteo total
        "Preprocessing": [],
        "Inference": [],
        "Postprocessing": []
    }

    for arch in archs:
        if cfg.device == 'cuda':
            torch.cuda.empty_cache()

        yolo = YOLO(arch+'.yaml', task='detect')
        n_l, n_p, n_g, flops = yolo.info()
        if cfg.onnx:
            yolo.export(
                format='onnx',
                imgsz=cfg.imgsz,
                opset=12,
                dynamic=False,
                simplify=True,
                project='./data/arch_benchmark/models',
            )
            yolo = YOLO('./data/arch_benchmark/models/' + arch + '.onnx', task='detect')
        t_pre, t_infer, t_post = inference_yolo_hard(cfg, yolo)

        # Añadir resultados al diccionario
        results["Model"].append(arch)
        results["Parameters"].append(n_p)
        results["GFLOPs"].append(flops)
        results["Preprocessing"].append(t_pre)
        results["Inference"].append(t_infer)
        results["Postprocessing"].append(t_post)

    # Convertir el diccionario de resultados en un DataFrame de pandas
    df = pd.DataFrame(results)
    if cfg.save:
        df.to_csv('./data/arch_benchmark/results.csv', index=False)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute FLOPs of a model.')
    parser.add_argument('--device', type=str, default='cpu', help='device')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='size of the input image size. default is 224')
    parser.add_argument('--num-runs', type=int, default=105,
                        help='number of runs to compute average forward timing. default is 101')
    parser.add_argument('--num-warmup-runs', type=int, default=5,
                        help='number of warmup runs to avoid initial slow speed. default is 5')
    parser.add_argument('--onnx', type=bool, default=True, help='export to onnx')
    parser.add_argument('--save', type=bool, default=True, help='save the model')

    # Nombres de los modelos subidos, asumiendo que corresponden a los nombres de archivo sin la extensión
    path = '../ultralytics/cfg/models/v8/'
    # get all files in the path
    #model_names = [f.split('.')[0] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    model_names = [
        'yolov8s',
        'yolov8s-p2',
        'yolov8s-p234',
        'yolov8s-p2-c2fsum',
        'yolov8s-p2-late_decoup',
        'yolov8s-p2-late_decoupV2',
        'yolov8s-p2-pconv_neck',
        'yolov8s-p2-pconv_bone',
        'yolov8s-p2-pconvema_bone',
        'yolov8s-p2-pconvsimam_bone',
    ]

    args = parser.parse_args()
    cfg = args
    # merge args into cfg
    df = perform_benchmark(cfg, model_names)
