import numpy as np
import torch
import random
import os
import fnmatch
import glob
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool
import argparse


######################################
#  FOURIER TRANSFORM STYLE TRANSFER
######################################

def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha


def low_freq_mutate_np(amp_src, amp_trg, L=0.1):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src


def FDA_source_to_target_np(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg


class ImgFetcher:
    '''Returns image paths and cycles once all images have been returned.
    '''

    def __init__(self, abs_img_dir):
        '''
        Args:
            abs_img_dir: Absolute path to image directory root.
        '''

        self.img_dir = abs_img_dir
        self.img_path_list = self.gen_img_path_list()


    def gen_img_path_list(self, ordered=False):
        '''Generates a list with .png image paths by crawling
        '''       
        img_paths = []
        for root, _, files in os.walk(self.img_dir):
            for filename in fnmatch.filter(files, "*.png"):
                img_paths.append(os.path.join(root, filename))

        if ordered:
            img_paths.sort()
        else:
            random.shuffle(img_paths)

        return img_paths


    def get_img_path(self):

        img_path = self.img_path_list.pop()

        if len(self.img_path_list) == 0:
            self.img_path_list = self.gen_img_path_list()
        
        return img_path


def read_image(img_path):    
    img = Image.open(img_path).convert('RGB')
    return img


def resize_img_cityscapes_to_a2d2(img):
    # Resize (W, H) : (2048, 1024) --> (2416, 1208) (aspect ratio const.)
    # Crop center : (2416, 1208) --> (1920, 1208)
    img = img.resize((2416, 1208), Image.ANTIALIAS)

    left = int((2416 - 1920)/2)
    right = int((2416 + 1920)/2)
    top = 0
    bottom = 1208
    img = img.crop((left, top, right, bottom))

    return img


def resize_img_a2d2_to_cityscapes(img):
    # Resize (W, H) : (1920, 1208) --> (2048, 1288) (aspect ratio const.)
    # Crop center : (2048, 1288) --> (2048, 1024)
    img = img.resize((2048, 1288), Image.ANTIALIAS)

    left = 0
    right = 2048
    top = int((1288 - 1024)/2)
    bottom = top + 1024
    img = img.crop((left, top, right, bottom))

    return img


def transform_img_source2target(src_img, trg_img, beta=0.01):
    '''
    Args:
        src_img: Numpy float array w. dim (H,W,C) in range (0., 255.)
        trg_img:

    Returns:
        PIL image
    '''
    
    src_img = np.asarray(src_img, np.float32)
    trg_img = np.asarray(trg_img, np.float32)

    src_img = src_img.transpose((2, 0, 1))
    trg_img = trg_img.transpose((2, 0, 1))

    transformed_src_img = FDA_source_to_target_np(src_img, trg_img, beta)

    transformed_src_img = transformed_src_img.transpose((1,2,0))

    transformed_src_img[transformed_src_img < 0.] = 0.
    transformed_src_img[transformed_src_img > 255.] = 255.

    transformed_src_img = Image.fromarray(transformed_src_img.astype(np.uint8))

    return transformed_src_img


def transform_img_cityscapes(sample):

    img_path, trg_img_path, dest_path, beta = sample

    src_img = read_image(img_path)
    trg_img = read_image(trg_img_path)

    trg_img = resize_img_cityscapes_to_a2d2(trg_img)

    trans_src_img = transform_img_source2target(src_img, trg_img, beta)

    img_filename = os.path.split(img_path)[-1]
    trans_src_img.save(os.path.join(dest_path, img_filename))


def transform_img_a2d2(sample):

    img_path, trg_img_path, dest_path, beta = sample

    src_img = read_image(img_path)
    trg_img = read_image(trg_img_path)

    trg_img = resize_img_a2d2_to_cityscapes(trg_img)

    trans_src_img = transform_img_source2target(src_img, trg_img, beta)

    img_filename = os.path.split(img_path)[-1]

    # Parse 'city' subdirectory
    idx = img_filename.index('_')
    city = img_filename[:idx]
    dest_path_city = os.path.join(dest_path, city)
    if os.path.isdir(dest_path_city) == False:
        os.mkdir(dest_path_city)

    trans_src_img.save(os.path.join(dest_path_city, img_filename))


def transfer_a2d2_to_cityscapes(a2d2_path, cityscapes_path, dest_path, beta=0.0015, nproc=1):
    '''
    Args:
        a2d2_path: Path to 'a2d2/.../img_dir/train' etc.
        cityscapes_path: Path to 'cityscapes/leftImg8bit/' etc.
        dest_path: Path to 'a2d2_trans/img_dir/train' etc.
    '''
    # Read all A2D2 image paths
    img_paths = []
    for root, _, files in os.walk(a2d2_path):
        for filename in fnmatch.filter(files, "*.png"):
            img_paths.append(os.path.join(root, filename))

    # Initialize target image fetcher
    img_fetcher = ImgFetcher(cityscapes_path)

    print("Arranging (src, trg) samples")
    samples = []
    N = len(img_paths)
    for idx, img_path in enumerate(img_paths):
        print(f"    Sample {idx}/{N} ({idx/N*100.}%)\r", end="")
        trg_img_path = img_fetcher.get_img_path()
        samples.append( (img_path, trg_img_path, dest_path, beta) )
    print(f"    Sample {N}/{N} ({(N)/N*100.}%)                ")

    print("Transforming images")
    with Pool(nproc) as p:
        p.map(transform_img_cityscapes, samples)


def transfer_cityscapes_to_a2d2(cityscapes_path, a2d2_path, dest_path, beta=0.0015, nproc=1):
    '''
    Args:
        cityscapes_path: Path to 'cityscapes/leftImg8bit/train/' etc.
        a2d2_path: Path to 'a2d2/.../img_dir/train' etc.
        dest_path: Path to 'cityscapes_trans/leftImg8bit/train/' etc.
    '''
    # Read all Cityscapes image paths
    img_paths = glob.glob(f'{cityscapes_path}/*/*leftImg8bit.png')

    # Initialize target image fetcher
    img_fetcher = ImgFetcher(a2d2_path)

    print("Arranging (src, trg) samples")
    samples = []
    N = len(img_paths)
    for idx, img_path in enumerate(img_paths):
        print(f"    Sample {idx}/{N} ({idx/N*100.}%)\r", end="")
        trg_img_path = img_fetcher.get_img_path()
        samples.append( (img_path, trg_img_path, dest_path, beta) )
    print(f"    Sample {N}/{N} ({(N)/N*100.}%)                ")

    print("Transforming images")
    with Pool(nproc) as p:
        p.map(transform_img_a2d2, samples)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert A2D2 <--> Cityscapes image style as low-level feature adaption')
    parser.add_argument(
        'src', type=str, help='a2d2 or cityscapes')
    parser.add_argument(
        'src_path', type=str, help='Example: a2d2/.../img_dir/train')
    parser.add_argument(
        'trg_path', type=str, help='Example: cityscapes/leftImg8bit/train')
    parser.add_argument(
        'out_path', type=str, help='Example: a2d2_trans/img_dir/train')
    parser.add_argument(
        '--beta', default=0.01, type=float,
        help='Beta value for FDA (A2D2: 0.0015, Cityscapes: 0.0015)')
    parser.add_argument('--nproc', default=1, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    src = args.src
    src_path = args.src_path
    trg_path = args.trg_path
    out_path = args.out_path
    beta = args.beta
    nproc = args.nproc

    if src == 'a2d2':
        transfer_a2d2_to_cityscapes(src_path, trg_path, out_path, beta, nproc)
    elif src == 'cityscapes':
        transfer_cityscapes_to_a2d2(src_path, trg_path, out_path, beta, nproc)
    else:
        raise Exception(f"Invalid source ({src})")

