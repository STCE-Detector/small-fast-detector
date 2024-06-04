import os
import cv2
import configparser

from tqdm import tqdm


def create_seq_info(seq_path, name, frame_rate, seq_length, im_width, im_height):
    config = configparser.ConfigParser()
    config['Sequence'] = {
        'name': name,
        'imDir': 'img1',
        'frameRate': str(frame_rate),
        'seqLength': str(seq_length),
        'imWidth': str(im_width),
        'imHeight': str(im_height),
        'imExt': '.jpg'
    }

    with open(os.path.join(seq_path, 'seqinfo.ini'), 'w') as configfile:
        config.write(configfile)


def process_videos(input_folder):
    output_folder = input_folder + "_seq"
    os.makedirs(output_folder, exist_ok=True)

    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".mp4"):
            video_path = os.path.join(input_folder, filename)
            video_name = os.path.splitext(filename)[0]
            seq_path = os.path.join(output_folder, video_name)
            img1_path = os.path.join(seq_path, 'img1')
            gt_path = os.path.join(seq_path, 'gt')

            os.makedirs(img1_path, exist_ok=True)
            os.makedirs(gt_path, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_id = 1

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_filename = f"{str(frame_id).zfill(6)}.jpg"
                cv2.imwrite(os.path.join(img1_path, frame_filename), frame)
                frame_id += 1

            seq_length = frame_id - 1
            create_seq_info(seq_path, video_name, frame_rate, seq_length, im_width, im_height)
            cap.release()

    print(f"Processing complete. Check the folder: {output_folder}")


# Replace 'your_folder_path' with the path to your folder containing .mp4 files
process_videos('/home-net/ierregue/project/track/general_selection')
