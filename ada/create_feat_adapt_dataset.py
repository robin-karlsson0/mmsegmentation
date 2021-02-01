import os
import glob


feat_adapt_dataset_path = "/media/robin/Data/feat_adapt_dataset"  #"var/dataset/feat_adapt_dataset/"
os.mkdir(feat_adapt_dataset_path)

# Root paths
a2d2_img_path = "/media/robin/Data/a2d2/camera_lidar_semantic/img_dir"  #"/var/datasets/a2d2/camera_lidar_semantic/img_dir"

cityscapes_img_path = "/media/robin/Data/cityscapes_dataset/leftImg8bit"  #"/var/datasets/cityscapes/leftImg8bit"

# Split paths
a2d2_img_train_path = os.path.join(a2d2_img_path, "train")
a2d2_img_val_path = os.path.join(a2d2_img_path, "val")
a2d2_img_test_path = os.path.join(a2d2_img_path, "test")

cityscapes_img_train_path = os.path.join(cityscapes_img_path, "train")
cityscapes_img_val_path = os.path.join(cityscapes_img_path, "val")
cityscapes_img_test_path = os.path.join(cityscapes_img_path, "test")

# Sample folders
a2d2_folders = []
a2d2_folders.append(a2d2_img_train_path)
a2d2_folders.append(a2d2_img_val_path)
a2d2_folders.append(a2d2_img_test_path)

cityscapes_folders = []
cityscapes_folders.append(cityscapes_img_train_path)
cityscapes_folders.append(cityscapes_img_val_path)
cityscapes_folders.append(cityscapes_img_test_path)

# Create directory structure
feat_adapt_a2d2_path = os.path.join(feat_adapt_dataset_path, 'a2d2')
feat_adapt_cityscapes_path = os.path.join(feat_adapt_dataset_path, 'cityscapes')
os.mkdir(feat_adapt_a2d2_path)
os.mkdir(feat_adapt_cityscapes_path)

# Symbolically link files 

idx = 0
for a2d2_folder in a2d2_folders:

    img_files = glob.glob(f"{a2d2_folder}/*.png")

    for img_file in img_files:

        link_file = os.path.join(feat_adapt_a2d2_path, f"{idx}.png")
        os.system(f"ln -s {img_file} {link_file}")
        idx += 1

print(f"Linked {idx} A2D2 images")

idx = 0
for cityscapes_folder in cityscapes_folders:
    
    img_files = glob.glob(f"{cityscapes_folder}/*/*.png")

    for img_file in img_files:

        link_file = os.path.join(feat_adapt_cityscapes_path, f"{idx}.png")
        os.system(f"ln -s {img_file} {link_file}")
        idx += 1

print(f"Linked {idx} Cityscapes images")