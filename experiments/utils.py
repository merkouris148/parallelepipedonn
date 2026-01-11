import os

def make_sample_filename_pfx(path_header, data_set_name, input_dir, class_identifier):
    tokens  = [path_header, data_set_name, input_dir, str(class_identifier)]
    path    = os.path.join(*tokens)
    path    += "-"

    return path

def make_dataset_dir_path(path_header, data_set_name):
    tokens = [path_header, data_set_name]
    path = os.path.join(*tokens)

    return path


def make_inputs_dir_path(path_header, data_set_name, input_dir):
    tokens = [path_header, data_set_name, input_dir]
    path = os.path.join(*tokens)

    return path


def make_outputs_dir_path(path_header, data_set_name, output_dir):
    tokens = [path_header, data_set_name, output_dir]
    path = os.path.join(*tokens)

    return path


def make_predictions_path(path_header, data_set_name, input_dir, preds_filename):

    tokens = [path_header, data_set_name, input_dir, preds_filename]
    path = os.path.join(*tokens)

    return path