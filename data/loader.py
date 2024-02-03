import datasets
from pathlib import Path

def load_ds(ds_name: str):
    """
    Load a dataset from disk or from the Hugging Face Hub.

    Parameters:
    - ds_name (str): The name of the dataset to load.

    Returns:
        - ds (datasets.Dataset): The loaded dataset

    Notes:
    - This function first checks if the dataset is already stored on disk. If it is, it loads the dataset from disk
    - If the dataset is not found on disk, it loads the dataset from the Hugging Face Hub and saves the dataset to disk for future loading"""
   
    ds_path =  Path(__file__).parent / ds_name
    try:
        ds = datasets.load_from_disk(ds_path)
    except FileNotFoundError:
        ds = datasets.load_dataset(ds_name)
        ds_path.mkdir(parents=True)
        ds.save_to_disk(ds_path)
    return ds
