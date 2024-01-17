import os
import shutil

# CREATE NEW DATASET STRUCTURE
root_path = '/data-fast/127-data2/ierregue/datasets'
#root_path = '/Users/inaki-eab/Desktop/DETECTOR_DATASETS'

dataset_1 = 'custom_no_coco_v3'
dataset_2 = 'refined_custom_coco'

new_dataset = 'custom_dataset_v2'
new_dataset_root = os.path.join(root_path, new_dataset)

# Create folder structure
if not os.path.isdir(new_dataset_root):
    os.makedirs(new_dataset_root)
    os.makedirs(os.path.join(new_dataset_root, 'labels','train'))
    os.makedirs(os.path.join(new_dataset_root, 'labels','test'))
    os.makedirs(os.path.join(new_dataset_root, 'labels','val'))
    os.makedirs(os.path.join(new_dataset_root, 'images','train'))
    os.makedirs(os.path.join(new_dataset_root, 'images','test'))
    os.makedirs(os.path.join(new_dataset_root, 'images','val'))


# CHECK IF DATASET 1 AND 2 HAVE COMMON FILES
d1_filenames = os.listdir(os.path.join(root_path,dataset_1,'labels','val'))
d2_filenames = os.listdir(os.path.join(root_path,dataset_2,'labels','val'))
common_filenames = list(set(d1_filenames) & set(d2_filenames))
assert len(common_filenames)==0, common_filenames

d1_filenames = os.listdir(os.path.join(root_path,dataset_1,'labels','train'))
d2_filenames = os.listdir(os.path.join(root_path,dataset_2,'labels','train'))
common_filenames = list(set(d1_filenames) & set(d2_filenames))
assert len(common_filenames)==0, common_filenames


# MERGE DATASET 1 AND 2
# Copy dataset 1 into new one
shutil.copytree(os.path.join(root_path,dataset_1),
                new_dataset_root,
                dirs_exist_ok=True)
# Copy dataset 2 into new one
shutil.copytree(os.path.join(root_path,dataset_2),
                new_dataset_root,
                dirs_exist_ok=True)


# check that the total num os elements corresponds to the sum of datasets
assert (len(os.listdir(os.path.join(root_path,dataset_1,'labels','val'))) +
        len(os.listdir(os.path.join(root_path,dataset_2,'labels','val'))) ==
        len(os.listdir(os.path.join(new_dataset_root,'labels','val'))))

assert (len(os.listdir(os.path.join(root_path,dataset_1,'images','val'))) +
        len(os.listdir(os.path.join(root_path,dataset_2,'images','val'))) ==
        len(os.listdir(os.path.join(new_dataset_root,'images','val'))))

assert (len(os.listdir(os.path.join(root_path,dataset_1,'images','train'))) +
        len(os.listdir(os.path.join(root_path,dataset_2,'images','train'))) ==
        len(os.listdir(os.path.join(new_dataset_root,'images','train'))))

assert (len(os.listdir(os.path.join(root_path,dataset_1,'labels','train'))) +
        len(os.listdir(os.path.join(root_path,dataset_2,'labels','train'))) ==
        len(os.listdir(os.path.join(new_dataset_root,'labels','train'))))
