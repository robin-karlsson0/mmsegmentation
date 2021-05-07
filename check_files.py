import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('a2d2_path', type=str)
parser.add_argument('--sample-dirs', nargs='+', type=str, required=True)
args = parser.parse_args()

a2d2_path = args.a2d2_path

for sample_dir in args.sample_dirs:

    for splice in ['train', 'val', 'test']:

        splice_path = os.path.join(a2d2_path, sample_dir, splice)

        subdirs = os.listdir(splice_path)
        subdirs_N = len(subdirs)

        for subdir in subdirs:

            subdir_path = os.path.join(splice_path, subdir)
            
            file_paths = glob.glob(os.path.join(subdir_path, '*.png'))

            file_idx = 0
            for file_path in file_paths:
                if file_idx % 100 == 0:
                    print(f"{sample_dir} | {splice} | subdir: {subdir}/{subdirs_N} ({int(subdir)/subdirs_N*100.:.0f}%) | file: {file_idx}\r", end="")
                os.system(f'pngcheck -q {file_path}')
                file_idx += 1
