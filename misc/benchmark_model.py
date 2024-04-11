import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from thop import profile, clever_format

from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def get_fps_latency(cfg, model):
    """
    Purpose: To measure the latency (inference time per image) and frames per second (FPS) of the model in a practical scenario.
    How It Works: After warming up the model, it runs a set number of inferences and calculates the total time taken. It then computes the average latency per inference and the corresponding FPS. This gives a direct indication of the model's performance in real-time applications.
    """
    model.eval()
    model.to(cfg.device)
    input_tensor = torch.autograd.Variable(torch.rand(cfg.bs, cfg.channels, cfg.imgsz, cfg.imgsz)).to(cfg.device)
    print('Model is loaded, start forwarding.')

    start_time = time.time()
    with torch.no_grad():
        for _ in range(cfg.num_runs - cfg.num_warmup_runs):
            pred = model(input_tensor)
    end_time = time.time()

    total_forward = end_time - start_time
    actual_num_runs = cfg.num_runs - cfg.num_warmup_runs
    latency = total_forward / actual_num_runs  # Latency per inference
    fps = actual_num_runs / total_forward  # Frames per second

    return total_forward, fps, latency


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


def inference_yolo_hard(cfg, yolo, config, num_images=100):
    """
    Purpose: The core function that orchestrates the entire benchmarking process for a list of model architectures.
    How It Works: For each architecture, it performs a series of measurements (like parameter count, FLOPs, inference time, FPS, and latency) using the functions described above. It aggregates these metrics into a comprehensive report (in the form of a DataFrame), allowing for an in-depth comparison of model performances.
    """
    speeds = []
    inference_list = []
    if cfg.device == 'cuda':
        device = torch.device('cuda')
    elif cfg.device == 'mps':
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    for _ in range(num_images):
        # Generate a random image
        if config.get('args', {}).get('half', False):
            image = torch.rand(1, 3, 640, 640).half().to(device)
        elif config.get('args', {}).get('int8', False):
            image = torch.rand(1, 3, 640, 640).half().to(device)
        else:
            image = torch.rand(1, 3, 640, 640).to(device)

        # Run inference
        results = yolo.predict(image, device=0, half=config.get('args', {}).get('half', False), int8=config.get('args', {}).get('int8', False))

        # Collect inference speed
        speeds.append(results[0].speed['inference'] + results[0].speed['postprocess'])
        inference_list.append(results[0].speed['inference'])

    # Calculate the median inference speed
    median_speed = np.median(speeds)
    median_inference = np.median(inference_list)
    median_speed_s = median_speed / 1000

    # Calculate median FPS (frames per second)
    median_fps = 1 / median_speed_s

    return median_speed, median_fps, median_inference


def perform_benchmark(cfg, archs, path='../ultralytics/cfg/models/v8/'):
    results = {
        "model": [],
        "parameters (count)": [],  # Sin unidad específica, conteo de parámetros
        "GFLOPs": [],  # Giga Floating Point Operations, sin unidad de tiempo porque es un conteo total
        "latency (ms)": [],  # Milisegundos
        "inference (ms)": [],
        "FPS (frames/s)": [],  # Frames (cuadros) por segundo
    }
    """export_configs = [
        {'format': 'pytorch', 'args': {'half': False}},
        {'format': 'pytorch', 'args': {'half': True}},
        {'format': 'torchscript', 'args': {'imgsz': cfg.imgsz, 'optimize': False}},
        {'format': 'onnx', 'args': {'imgsz': cfg.imgsz, 'half': False, 'dynamic': False, 'int8': False, 'simplify': False, 'opset': 12}},
        {'format': 'onnx', 'args': {'imgsz': cfg.imgsz, 'half': False, 'dynamic': False, 'int8': False, 'simplify': True, 'opset': 12}},
        {'format': 'onnx', 'args': {'imgsz': cfg.imgsz, 'half': True, 'dynamic': False, 'int8': False, 'simplify': True, 'opset': 12}},
        {'format': 'onnx', 'args': {'imgsz': cfg.imgsz, 'half': False, 'dynamic': False, 'int8': True, 'simplify': True, 'opset': 12}},
        {'format': 'engine', 'args': {'imgsz': cfg.imgsz, 'half': False, 'dynamic': False, 'simplify': False, 'workspace': 4}},
        {'format': 'engine', 'args': {'imgsz': cfg.imgsz, 'half': False, 'dynamic': False, 'simplify': True, 'workspace': 4}},
        {'format': 'engine', 'args': {'imgsz': cfg.imgsz, 'half': True, 'dynamic': False, 'simplify': True, 'workspace': 4}},
    ]
"""
    export_configs = [
        {'format': 'pytorch', 'args': {'half': False}},
    ]
    for arch in archs:
        full_path = check_file_exists(path, arch)

        for config in export_configs:
            if cfg.device == 'cuda':
                torch.cuda.empty_cache()

            yolo = YOLO(full_path, task='detect')
            yolo.to(cfg.device)
            model = yolo.model
            n_l, n_p, n_g, flops = model_info(model)

            try:
                args_str = json.dumps(config['args'], sort_keys=True)
                export_filename = f"{arch}.{config['format']}"
                export_path = f'./{export_filename}'
                unique_id = ''.join(e for e in args_str if e.isalnum())
                export_filename = f"{arch}_{config['format']}_{unique_id}"
                if config['format'] == 'pytorch':
                    pass
                else:
                    yolo.export(format=config['format'], device=cfg.device, **config['args'])
                    print(f"Modelo {arch} exportado como {export_filename} a {export_path}")
                    model_path = export_path if os.path.exists(export_path) else f"{export_path}.{config['format']}"
                    yolo = YOLO(model_path, task='detect')
            except Exception as e:
                print(f"Error exporting model {arch} to format {config['format']}: {e}")
                continue
            inference_time_yolo_hard, fps_yolo, inference_ms = inference_yolo_hard(cfg, yolo, config)

            # Añadir resultados al diccionario
            results["model"].append(export_filename)
            results["parameters (count)"].append("{:,.0f}".format(n_p))
            results["GFLOPs"].append("{:,.2f}".format(flops))
            results["latency (ms)"].append("{:.2f}".format(inference_time_yolo_hard))
            results["inference (ms)"].append("{:.2f}".format(inference_ms))
            results["FPS (frames/s)"].append("{:.2f}".format(fps_yolo))

    # Convertir el diccionario de resultados en un DataFrame de pandas
    df = pd.DataFrame(results)
    # plot_results(df)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute FLOPs of a model.')
    parser.add_argument('--bs', type=int, default=1, help='batch size')
    parser.add_argument('--channels', type=int, default=3, help='batch size')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--num-frames', type=int, default=32, help='temporal clip length.')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='size of the input image size. default is 224')
    parser.add_argument('--num-runs', type=int, default=105,
                        help='number of runs to compute average forward timing. default is 105')
    parser.add_argument('--num-warmup-runs', type=int, default=5,
                        help='number of warmup runs to avoid initial slow speed. default is 5')

    # Nombres de los modelos subidos, asumiendo que corresponden a los nombres de archivo sin la extensión
    path = '../ultralytics/cfg/models/v8/'
    # get all files in the path
    model_names = [f.split('.')[0] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    model_names = sorted(model_names)

    args = parser.parse_args()
    cfg = args
    # merge args into cfg
    df = perform_benchmark(cfg, model_names)
    df.to_csv('benchmark_results.csv')
