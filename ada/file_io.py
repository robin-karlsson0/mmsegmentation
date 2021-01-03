#!/usr/bin/env python
import pickle
import gzip
import os


def write_compressed_pickle(obj, filename, write_dir):
    '''Converts an object into byte representation and writes a compressed file.

    Args:
        obj: Generic Python object.
        filename: Name of file without file ending.
        write_dir (str): Output path.
    '''
    path = os.path.join(write_dir, f"{filename}.gz")
    pkl_obj = pickle.dumps(obj)
    try:
        with gzip.open(path, "wb") as f:
            f.write(pkl_obj)
    except IOError as error:
        print(error)


def read_compressed_pickle(file_path):
    '''Reads a compressed binary file and return the object.

    Args:
        file_path (str): Path to the file (incl. filename)
    '''
    try:
        with gzip.open(file_path, "rb") as f:
            pkl_obj = f.read()
            obj = pickle.loads(pkl_obj)
            return obj
    except IOError as error:
        print(error)