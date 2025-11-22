import os
import shutil

def reset_directory(path):
    """
    If the directory exists, delete all its contents.
    If it doesn't exist, create it.
    """
    if os.path.exists(path):
        # Remove all contents inside the directory
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
    else:
        # Create the directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

def create_directory(path):
    os.makedirs(path, exist_ok=True)
