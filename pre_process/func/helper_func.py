import os

def get_file_lst(input_path, extension):
    """
    return file list
    """
    file_lst = []
    for path, subdirs, files in os.walk(input_path):
        for name in files:
            file_candidate = os.path.join(path, name)
            if (extension in file_candidate):
                file_lst.append(file_candidate)
    return file_lst