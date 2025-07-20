import logging, joblib, io
import pandas as pd
from   s3fs   import S3FileSystem

def pd_info(df: pd.DataFrame):
    with io.StringIO() as output:
        df.info(show_counts=True, buf=output)
        return output.getvalue()

def load_csv_files(s3: S3FileSystem, path_list: list):
    try:
        df = pd.DataFrame()
        for path in path_list:
            with s3.open(path, mode='r') as fd:
                df = pd.concat([df, pd.read_csv(fd)], axis=0, ignore_index=True)
        return df
    except:
        print(path)
        return None

def delete_s3_files(s3: S3FileSystem, files: dict):
    try:
        for k,path in files.items():
            s3.rm(path)
            print(f"rm {path}")
    except:
        print(path)
        pass
    return

def load_parquet_file(s3: S3FileSystem, path: str, logger=logging.getLogger(), verbose=True):
    if verbose:  logger.info(f"Loading file: {path}")
    try:
        df = pd.DataFrame()
        with s3.open(path, mode='rb') as fd:
            df = pd.read_parquet(fd)
        return df
    except:
        logger.error(f"Error loading file: {path}")
        return None

def save_to_parquet(df, s3: S3FileSystem, path: str, verbose=True):
    with s3.open(path, mode='wb') as fd:
        df.to_parquet(fd)
    if verbose:  logging.info(f"\n{path}\n{pd_info(df)}")
    return

def save_to_pkl(obj, s3: S3FileSystem, path: str):
    with s3.open(path, mode='wb') as fd:
        joblib.dump (obj, fd)
    return

def load_pkl_file(s3: S3FileSystem, path: str):
    try:
        with s3.open(path, mode='rb') as fd:
            obj = joblib.load(fd)
        return obj
    except:
        logging.error(f"Error loading file: {path}")
        return None
