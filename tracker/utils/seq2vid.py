import configparser
import os

import cv2
from tqdm import tqdm


def seq2vid(dataset_root, seq, output_folder):
    seq_path = os.path.join(dataset_root, seq)
    images_folder = os.path.join(seq_path, 'img1')
    output_video_path = os.path.join(output_folder, seq + '.mp4')

    # Read the sequence info
    seqinfo_file = os.path.join(seq_path, 'seqinfo.ini')
    config = configparser.ConfigParser()
    config.read(seqinfo_file)
    fps = int(float(config['Sequence']['frameRate']))

    # Get the list of images
    images = [img for img in os.listdir(images_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()  # Ensure the images are in the correct order

    # Read the first image to get the dimensions
    first_image_path = os.path.join(images_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec if needed
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Iterate through images and write them to the video
    for image in images:
        img_path = os.path.join(images_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # Release the VideoWriter object
    video.release()


if __name__ == '__main__':
    dataset_name = 'CTD'

    dataset_root = './../evaluation/TrackEval/data/gt/mot_challenge/' + dataset_name
    output_folder = './../../data/' + dataset_name

    # Create the output folder
    os.makedirs(output_folder, exist_ok=True)
    # Read sequences from the dataset root
    sequences = [seq for seq in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, seq))]
    # Process each sequence
    for seq in tqdm(sequences):
        seq2vid(dataset_root, seq, output_folder)

