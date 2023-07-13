import json
import warnings
from typing import Dict, List, Union

import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import pipeline

nltk.download("punkt")
warnings.filterwarnings("ignore")


def check_for_empty_text_fields(
    dataframes: List[pd.DataFrame], text_fields: List[str]
) -> None:
    """
    Check for empty text fields in a list of DataFrames and print the results.

    Args:
        dataframes (List[pd.DataFrame]): A list of DataFrames to check for empty text fields.
        text_fields (List[str]): A list of column names representing the text fields to check.

    Returns:
        None

    Prints:
        - If empty text fields are found, prints the DataFrame index and corresponding
        text field names where empty fields are present.
        - If no empty text fields are found, prints "No empty text fields found."
    """
    result = False
    empty_fields = []
    for idx, df in enumerate(dataframes):
        for text_field in text_fields:
            # check for empty rows
            if df[text_field].isna().any():
                result = True
                empty_fields.append((f"Dataframe {idx}", text_field))

    if result:
        print(f"There are empty text fields. Found here: {empty_fields}")
    else:
        print(0)

    return None


def get_dfs_for_exploration(dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Takes a list of DataFrames and calculates the length of the text fields and
    the reduction in length between the argument and conclusion.

    Args:
        dataframes (List[pd.DataFrame]): A list of DataFrames to be processed.

    Returns:
        List[pd.DataFrame]: A list of processed DataFrames with additional columns added.
    """
    new_dfs = []
    for df in dataframes:
        df["argument_len"] = df["argument"].str.lower()
        df["conclusion_len"] = df["conclusion"].str.lower()

        # calculate length of text fields
        df["argument_len"] = df["argument"].apply(len)
        df["conclusion_len"] = df["conclusion"].apply(len)

        # calculate reduction in length between argument and conclusion
        df["length_reduction"] = abs(df["conclusion_len"] / df["argument_len"] - 1)

        # count the number of sentences in the argument and conclusion
        df["arg_sentences"] = df["argument"].apply(lambda x: nltk.sent_tokenize(x))
        df["con_sentences"] = df["conclusion"].apply(lambda x: nltk.sent_tokenize(x))
        df["arg_sentence_count"] = df["arg_sentences"].apply(lambda x: len(x))
        df["con_sentence_count"] = df["con_sentences"].apply(lambda x: len(x))

        new_dfs.append(df)

    return new_dfs


def add_features_to_df(
    dataframes: Union[pd.DataFrame, List[pd.DataFrame]],
    model: SentenceTransformer,
    random_state: int = 1337,
) -> List[pd.DataFrame]:
    """
    Adds additional features to each DataFrame in the given list or a single DataFrame.
    - list of words
    - list of sentences
    - sentence embeddings
    - random sentence from argument
    - most similar sentence to conclusion
    - least similar sentence to conclusion

    Args:
        dataframes (Union[pd.DataFrame, List[pd.DataFrame]]): A single DataFrame or a list of DataFrames.
        model (SentenceTransformer): The SentenceTransformer model used for computing sentence embeddings.
        random_state (int, optional): Random seed.

    Returns:
        List[pd.DataFrame]: Modified DataFrame.
    """
    # handle single df
    if isinstance(dataframes, pd.DataFrame):
        dataframes = [dataframes]

    # set random seed
    np.random.seed(random_state)

    return [compute_features(df, model) for df in tqdm(dataframes)]


def compute_features(df: pd.DataFrame, model: SentenceTransformer) -> pd.DataFrame:
    """
    Computes additional features for a given DataFrame using a SentenceTransformer model.

    Args:
        df (pd.DataFrame): You're not reading all this, right?
        model (SentenceTransformer): The model used for computing sentence embeddings.

    Returns:
        pd.DataFrame: The modified DataFrame with additional features.
    """
    # get list of words nltk.word_tokenize
    df["arg_words"] = df["argument"].apply(nltk.word_tokenize)
    df["con_words"] = df["conclusion"].apply(nltk.word_tokenize)

    # get list of sentences
    df["arg_sent"] = df["argument"].apply(nltk.sent_tokenize)
    df["arg_sent_count"] = df["arg_sent"].apply(len)
    df["con_sent"] = df["conclusion"].apply(nltk.sent_tokenize)
    df["con_sent_count"] = df["con_sent"].apply(len)

    # flatten lists
    arg_lengths = df["arg_sent"].apply(len)
    con_lengths = df["con_sent"].apply(len)
    arg_flat = [sent for sublist in df["arg_sent"].tolist() for sent in sublist]
    con_flat = [sent for sublist in df["con_sent"].tolist() for sent in sublist]

    # get embeddings and reconstruct flattened list
    # compute embeddings on list instead of rows to speed up process
    arg_emb_flat = model.encode(arg_flat)
    con_emb_flat = model.encode(con_flat)
    df["arg_sent_emb"] = reconstruct_list(arg_emb_flat, arg_lengths)
    df["con_sent_emb"] = reconstruct_list(con_emb_flat, con_lengths)

    # find least similar sentence to conclusion
    df["least_similar"] = df.apply(
        lambda row: np.argmin(
            cosine_similarity([row["con_sent_emb"][0]], row["arg_sent_emb"])
        ),
        axis=1,
    )

    # find most similar sentence to conclusion
    df["most_similar"] = df.apply(
        lambda row: np.argmax(
            cosine_similarity([row["con_sent_emb"][0]], row["arg_sent_emb"])
        ),
        axis=1,
    )

    # get random sentence from arguments
    # compute embeddings on list instead of rows to speed up process
    df["random_sent"] = df["arg_sent"].apply(np.random.choice)
    df["random_sent_emb"] = model.encode(
        df["random_sent"].tolist()
    ).tolist()  # I don't know why I need to do this, but it works

    return df


def reconstruct_list(flat_list: List[float], lengths: List[int]) -> List[List[float]]:
    """
    Reconstructs a nested list from a flat list and corresponding lengths.

    Args:
        flat_list (List[float]): A flat list of values.
        lengths (List[int]): A list of lengths specifying the number of elements in each sub-list.

    Returns:
        List[List[float]]: A nested list constructed from the flat list, where each sub-list has a length specified by 'lengths'.
    """
    emb_list = []
    i = 0
    for length in lengths:
        emb_list.append(flat_list[i : i + length])
        i += length
    return emb_list


def build_output(df: pd.DataFrame) -> Dict[str, str]:
    """
    Constructs a dict with the 'id' and 'predicted_conclusion' columns
    from the given DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with 'id' and 'predicted_conclusion' columns.

    Returns:
        Dict[str, str]: A dict with key-value pair from the DataFrame. Key='id', value='predicted_conclusion'.
    """
    return {row["id"]: row["predicted_conclusion"] for _, row in df.iterrows()}


def get_eval_command(
    result: Dict[str, str], data_path: str, output_path: str, file_name: str, split: str
) -> str:
    """
    Build evaluation command for given script to compare true and predicted values.

    Args:
        true_path (str): Path to the file containing the true values.
        pred_path (str): Path to the file containing my predicted values.

    Returns:
        str: Command to run in terminal to evaluate the predictions.
    """
    # get paths
    true_path = f"{data_path}{split}_data.json"
    pred_path = f"{output_path}{file_name}-{split}.json"

    # write predictions to file
    with open(pred_path, "w") as f:
        json.dump(result, f)

    return f"python eval.py --true {true_path} --predictions {pred_path}"


def get_BLEU_scores(pred_data, true_path: str) -> Dict[str, float]:
    """Just copied from the eval script, but without the print statement."""
    with open(true_path, "r") as f:
        true_data = json.load(f)

    # Unpack all sentences into a dict where the id is the key
    # and the value the text of the sentence
    true_ids = []
    true_conclusions = []
    for item in true_data:
        true_ids.append(item["id"])
        true_conclusions.append(item["conclusion"])

    pred_conclusions = [pred_data[i] for i in true_ids]
    pred_conclusions = [word_tokenize(c) for c in pred_conclusions]
    true_conclusions = [[word_tokenize(c)] for c in true_conclusions]

    # Calculate actual score
    bleu1_score = sum(
        sentence_bleu(true_c, pred_c, weights=(1, 0, 0, 0))
        for true_c, pred_c in zip(true_conclusions, pred_conclusions)
    ) / len(true_conclusions)
    bleu2_score = sum(
        sentence_bleu(true_c, pred_c, weights=(0, 1, 0, 0))
        for true_c, pred_c in zip(true_conclusions, pred_conclusions)
    ) / len(true_conclusions)
    bleu_score = sum(
        sentence_bleu(true_c, pred_c)
        for true_c, pred_c in zip(true_conclusions, pred_conclusions)
    ) / len(true_conclusions)

    return {"bleu_1": bleu1_score, "bleu_2": bleu2_score, "bleu": bleu_score}
