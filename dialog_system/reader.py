import numpy as np
import pickle


def load_embs(file):
    """Binary file loader for embeddings

    Args:
        file: absolute file path
    Returns:
        embs: list of embeddings
    """
    with open(file, "rb") as f:
        embs = pickle.load(f)
    return embs


def get_data(df, embeddings, rel2idx, subset=None, random=False):
    """Formats the data into a format for training

    Args:
        df: pd.DataFrame of input questions and relations
        embeddings: list of embeddings for each instance in df
        rel2idx: dict, a mapping of relation to index
        subset: int, the number of instances to get from the data
        random: bool, type of embeddings, random has only has shape (768, number of instances), whereas non-random has
            shape (768, number of instances, 11)
    Returns:
        A tuple of arrays of embeddings, relations, and indices
    """
    assert len(df.tokens) == len(embeddings)
    assert len(df.tokens) == len(df.relation)

    sub = len(df.relation)
    if subset:
        sub = subset

    embs = []
    rels = []
    idxs = []

    for i in range(len(df.relation))[:sub]:
        if random:
            cls_emb = embeddings[i]
        else:
            cls_emb = embeddings[i][0]
        embs.append(np.asarray(cls_emb, dtype=np.float32))
        rels.append(rel2idx[df.relation[i]])
        idxs.append(df.id[i])

    #     return data  # (emb (768,), rel, id)
    return (
        np.asarray(embs),
        np.asarray(rels, dtype=np.int16),
        np.asarray(idxs, dtype=np.int32),
    )
