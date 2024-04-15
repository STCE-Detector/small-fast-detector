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


def measure_inference_time(cfg, model, repetitions=300, warmup_iterations=10):
    """
    Measure inference time of a model.
    Purpose: This function measures the average inference time and its standard deviation of a model over a specified number of repetitions. It performs a "warm-up" phase to ensure the measurements are stable and not affected by initial caching/loading behaviors.
    How It Works: It first prepares a dummy input image (or batch of images) according to the specified configuration (e.g., batch size, image size). It then runs this input through the model multiple times, measuring how long each inference takes. If running on a GPU, it uses CUDA events for accurate timing; otherwise, it falls back to using the system time. It returns the average time and standard deviation across all repetitions.
    """
    device = torch.device(cfg.device)
    model.to(device)
    model.eval()
    # Ajustando para una sola imagen o un batch de imágenes con dimensiones [B, C, H, W]
    dummy_input = torch.rand(cfg.bs, cfg.channels, cfg.imgsz, cfg.imgsz).to(device)

    # GPU-WARM-UP
    for _ in range(warmup_iterations):
        _ = model(dummy_input)

    timings = np.zeros(repetitions)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            if cfg.device == 'cuda':
                # Usar eventos CUDA para GPU
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()  # Esperar a que la GPU termine
                curr_time = starter.elapsed_time(ender)  # Milisegundos
            else:
                # Usar time.time para CPU y otros dispositivos
                start_time = time.time()
                _ = model(dummy_input)
                end_time = time.time()
                curr_time = (end_time - start_time) * 1000  # Convertir a milisegundos
            timings[rep] = curr_time

    mean_time = np.mean(timings)
    std_dev_time = np.std(timings)
    return mean_time, std_dev_time


def measure_throughput(cfg, model, repetitions=300):
    """
    Purpose: This function calculates the throughput of the model, which is a measure of how many images the model can process per unit of time (images per second).
    How It Works: Similar to measure_inference_time, it uses a dummy input to repeatedly run the model for a given number of repetitions. It then divides the total number of processed images by the total time taken to process them, providing a measure of throughput.
    """
    device = torch.device(cfg.device)
    model.to(device)
    model.eval()
    # Ajustando para un batch de imágenes
    dummy_input = torch.rand(cfg.bs, cfg.channels, cfg.imgsz, cfg.imgsz).to(device)

    start_time = time.time()
    with torch.no_grad():
        for _ in range(repetitions):
            _ = model(dummy_input)
    total_time = time.time() - start_time

    throughput = (cfg.bs * repetitions) / total_time
    return throughput


def get_parameters(model):
    """
    Purpose: This function calculates the total number of trainable parameters in the model. This is a measure of the model's complexity.
    How It Works: It iterates over all parameters of the model, sums their sizes, and returns the total count. This helps understand the model's size and potential computational demands.
    """
    return sum(p.numel() for p in model.parameters())


def get_flops(cfg, model):
    """
    Purpose: To estimate the number of floating-point operations (FLOPs) the model performs during inference. This is another measure of model complexity and computational intensity.
    How It Works: It uses a tool to profile the model with a dummy input, calculating the total FLOPs required to process it. This gives an idea of how computationally intensive the model is.
    """
    input_tensor = torch.autograd.Variable(torch.rand(cfg.bs, cfg.channels, cfg.imgsz, cfg.imgsz))

    macs, params = profile(model, inputs=(input_tensor,))
    macs, params = clever_format([macs, params], "%.3f")
    return macs


def get_flops_with_torch_profiler(model, imgsz=640):
    """
    Purpose: An alternative way to calculate FLOPs, potentially offering more detail or accuracy using PyTorch's built-in profiling tools.
    How It Works: This function also profiles the model with a dummy input but uses PyTorch's profiler. It adjusts the FLOPs calculation based on the input image size, providing a more precise measure tailored to the specific operational scenario.
    """
    # model = de_parallel(model) in case of pasing DDP model, it transforms it to a single GPU model
    p = next(model.parameters())
    stride = (max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32) * 2  # max stride
    im = torch.zeros((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
    with torch.profiler.profile(with_flops=True) as prof:
        model(im)
    flops = sum(x.flops for x in prof.key_averages()) / 1E9
    imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]  # expand if int/float
    flops = flops * imgsz[0] / stride * imgsz[1] / stride  # 640x640 GFLOPs
    return flops


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


def inference_yolo(cfg, yolo):
    """
    Purpose: A basic function to run inference with the YOLO model on a single image and return the inference speed.
    How It Works: It generates a random image and passes it through the YOLO model, collecting and returning the speed of inference.
    """
    results = yolo(torch.rand(1, 3, 640, 640).to(cfg.device))
    return results[0].speed['inference']


def inference_yolo_hard(cfg, yolo, config, num_test_images=100, num_warmup_images=10):
    """
    Orchestrate the benchmarking process for a YOLO model using a specific number of test images and additional warmup images to stabilize performance metrics.

    Args:
    cfg: Configuration object with device settings.
    yolo: YOLO model object.
    config: Configuration dictionary with additional arguments.
    num_test_images: Number of test images for gathering metrics, default is 100.
    num_warmup_images: Number of warmup images to prepare the model, default is 10.

    Returns:
    Tuple containing mean speed in milliseconds, frames per second, and mean inference time.
    """
    speeds = []
    inference_times = []

    # Set device based on configuration
    device_type = cfg.device if cfg.device in ['cuda', 'mps'] else 'cpu'
    device = torch.device(device_type)

    # Warmup phase
    for _ in range(num_warmup_images):
        image = torch.rand((1, 3, 640, 640),
                           dtype=torch.float16 if config.get('args', {}).get('half', False) else torch.float32,
                           device=device)
        _ = yolo.predict(image, device=cfg.device, half=config.get('args', {}).get('half', False),
                         int8=config.get('args', {}).get('int8', False))

    # Test phase
    for _ in range(num_test_images):
        image = torch.rand((1, 3, 640, 640),
                           dtype=torch.float16 if config.get('args', {}).get('half', False) else torch.float32,
                           device=device)
        results = yolo.predict(image, device=0, half=config.get('args', {}).get('half', False),
                               int8=config.get('args', {}).get('int8', False))

        total_speed = sum(results[
                              0].speed.values())  # Assuming results[0].speed contains 'preprocess', 'inference', 'postprocess' times
        speeds.append(total_speed)
        inference_times.append(results[0].speed['inference'])

    # Calculate mean speeds and convert to seconds for FPS calculation
    mean_speed = np.mean(speeds)
    mean_inference = np.mean(inference_times)
    fps = 1000 / mean_speed if mean_speed > 0 else 0  # Convert ms to s and calculate FPS

    return mean_speed, fps, mean_inference


def perform_benchmark(cfg, archs, path='../ultralytics/cfg/models/v8/'):
    results = {
        "model": [],
        "parameters (count)": [],  # Sin unidad específica, conteo de parámetros
        "GFLOPs": [],  # Giga Floating Point Operations, sin unidad de tiempo porque es un conteo total
        "latency (ms)": [],  # Milisegundos
        "inference (ms)": [],
        "FPS (frames/s)": [],  # Frames (cuadros) por segundo
    }
    export_configs = [
        {'format': 'pytorch', 'args': {'half': False}},
        {'format': 'pytorch', 'args': {'half': True}},
        {'format': 'torchscript', 'args': {'imgsz': cfg.imgsz, 'optimize': False}},
        {'format': 'onnx', 'args': {'imgsz': cfg.imgsz, 'half': False, 'dynamic': False, 'int8': False, 'simplify': False}},
        {'format': 'onnx', 'args': {'imgsz': cfg.imgsz, 'half': False, 'dynamic': False, 'int8': False, 'simplify': True}},
        {'format': 'onnx', 'args': {'imgsz': cfg.imgsz, 'half': True, 'dynamic': False, 'int8': False, 'simplify': True}},
        {'format': 'onnx', 'args': {'imgsz': cfg.imgsz, 'half': False, 'dynamic': False, 'int8': True, 'simplify': True}},
        {'format': 'engine', 'args': {'imgsz': cfg.imgsz, 'half': False, 'dynamic': False, 'int8': False, 'simplify': False, 'workspace': 4}},
        {'format': 'engine', 'args': {'imgsz': cfg.imgsz, 'half': False, 'dynamic': False, 'int8': False, 'simplify': True, 'workspace': 4}},
        {'format': 'engine', 'args': {'imgsz': cfg.imgsz, 'half': True, 'dynamic': False, 'int8': False, 'simplify': True, 'workspace': 4}},
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
                export_path = f'./models/{export_filename}'
                args_dict = json.loads(args_str)
                unique_id = '_'.join(f"{key}_{value}" for key, value in args_dict.items())
                export_filename = f"{arch}_{config['format']}_{unique_id}"
                if config['format'] == 'pytorch':
                    pass
                else:
                    yolo.export(format=config['format'], device=cfg.device, **config['args'], project='./models/')
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
    parser.add_argument('--device', type=str, default='mps', help='device')
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
