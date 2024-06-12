import os
import json
import cv2
import numpy as np

root_folder = '/Users/inaki-eab/Desktop/IR/anti-uav/test-dev'
target_folder = '/Users/inaki-eab/Desktop/IR/anti-uav/yolo'

sequences = os.listdir(root_folder)
for seq in sequences:
    seq_path = os.path.join(root_folder, seq)
    if os.path.isdir(seq_path):
        video_path = os.path.join(seq_path, 'IR.mp4')
        annots_path = os.path.join(seq_path, 'IR_label.json')

        target_images_path = os.path.join(target_folder, 'images/val')
        target_annot_images_path = os.path.join(target_folder, 'annot_images/val')
        target_labels_path = os.path.join(target_folder, 'labels/val')

        # Create the output folder if it doesn't exist
        if not os.path.exists(target_images_path):
            os.makedirs(target_images_path)

        if not os.path.exists(target_labels_path):
            os.makedirs(target_labels_path)

        if not os.path.exists(target_annot_images_path):
            os.makedirs(target_annot_images_path)

        # Load the annotations
        with open(annots_path, 'r') as file:
            raw_annots = json.load(file)


        ### Extract frames from video ###
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        vid_name = video_path.split('/')[-2]

        frame_count = 0
        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            # If the frame was not retrieved, break the loop
            if not ret:
                break

            # Construct the filename for the output image
            if frame_count % 100 == 0:
                fname = f"{vid_name}_{frame_count:04d}"
                frame_filename = os.path.join(target_images_path, fname + ".jpg")

                # Save the frame as a JPG image
                cv2.imwrite(frame_filename, frame)

                # Get the annotations for the current frame
                bboxes = raw_annots['gt_rect'][frame_count]

                if len(bboxes) == 0:
                    continue

                bbox = np.array(bboxes)

                # Convert format xt yt w h -> xc yc w h normalized
                xt, yt, w, h = bbox

                # Get the image dimensions
                image_height, image_width, _ = frame.shape

                # Calculate normalized coordinates
                xcn = (xt + w / 2) / image_width
                ycn = (yt + h / 2) / image_height
                wn = w / image_width
                hn = h / image_height

                # Create the label file
                array = np.array([3, xcn, ycn, wn, hn])

                np.savetxt(os.path.join(target_labels_path, fname + ".txt"), array.reshape(1,-1))

                # draw the bounding box
                cv2.rectangle(frame, (int(xt), int(yt)), (int(xt + w), int(yt + h)), (0, 255, 0), 2)
                # save the image in another folder
                frame_filename = os.path.join(target_annot_images_path, fname + ".jpg")
                cv2.imwrite(frame_filename, frame)

            # Increment the frame count
            frame_count += 1

        # Release the video capture object
        cap.release()