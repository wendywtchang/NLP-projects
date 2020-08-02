import numpy as np


def divide_chunks(data, n):
    """Divides the data into equal sized chunks

    Args:
        data: data to divide
        n: int, number of chunks to divide into
    Returns:
        A list of lists containing the data divided into chunks
    """
    idxs = [i for i in range(len(data[0]))]
    for i in range(0, len(idxs), n):
        yield idxs[i : i + n]


def split_into_batches(data, batch_size):
    """Splits the data into batches

    Args:
        data: data to split
        batch_size: int, size of each batch
    Returns:
        batched_data: list, a list of lists for embeddings, relations, and indices
    """
    batches = list(divide_chunks(data, batch_size))
    batched_data = []

    for batch_idxs in batches:
        embs = []
        rels = []
        idxs = []

        for i in batch_idxs:
            embs.append(data[0][i])
            rels.append(data[1][i])
            idxs.append(data[2][i])

        batched_data.append((np.asarray(embs), np.asarray(rels), idxs))
    return batched_data


def make_filename(config):
    """Creates a file name out of the parameters in a configuration file

    Args:
        config: configparser.ConfigParser, a configuration object
    Returns:
        filename: str, a file name
    """
    filename = ""
    for i, para in enumerate(config):
        sep = "&" if i > 0 else ""
        if "model_path" in para:
            continue
        filename += f"{sep}{para}={config[para]}"
    filename += ".pt"
    return filename


def get_log_params(config):
    """Creates a dictionary out of a configuration file

    Args:
        config: configparser.ConfigParser, a configuration object
    Returns:
        params: dict, a dictionary mapping parameter name and value
    """
    params = {}
    for sec in config:
        for k, v in config[sec].items():
            params[sec + "/" + k] = v
    return params
