from typing import Dict, List, Union

import nltk
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import pipeline

nltk.download("punkt")


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
        df (pd.DataFrame): Given DataFrame.
        model (SentenceTransformer): The model used for computing sentence embeddings.

    Returns:
        pd.DataFrame: The modified DataFrame with additional features.
    """
    # get list of sentences
    df["arg_sent"] = df["argument"].apply(nltk.sent_tokenize)
    df["con_sent"] = df["conclusion"].apply(nltk.sent_tokenize)

    # get list of words nltk.word_tokenize
    df["arg_words"] = df["argument"].apply(nltk.word_tokenize)
    df["con_words"] = df["conclusion"].apply(nltk.word_tokenize)

    # compute sentence embeddings
    df["arg_sent_emb"] = list(model.encode(df["arg_sent"].tolist()))
    df["con_sent_emb"] = list(model.encode(df["con_sent"].tolist()))

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

    # get random sentence from argument sentences
    df["random_sent"] = df["arg_sent"].apply(np.random.choice)

    return df


# def add_features_to_df_old(
#     dataframes: Union[pd.DataFrame, List[pd.DataFrame]],
#     model: SentenceTransformer = None,
#     columns: Union[List[str], str] = None,
#     use_pca: bool = True,
#     variance: float = 0.95,
#     random_state: int = 0,
#     sentiment: bool = False,
# ) -> List[pd.DataFrame]:
#     """
#     Adds additional features to each DataFrame in the given list.

#     Args:
#         dataframes (List[pd.DataFrame]): A list of DataFrames.

#     Returns:
#         List[pd.DataFrame]: A list of modified DataFrames where the following features
#         are added to each DataFrame:
#         - 'sentences': Contains a list of sentences from the 'argument' column.
#         - 'word_list': Contains a list of words from the 'argument' column.
#         - 'sent_embeddings': Contains the sentence embeddings computed using the provided BERT model.
#         - 'pca_sent_embeddings': Contains the PCA-transformed sentence embeddings, retaining 95% of variance.
#     """

#     # handle single df and column
#     if isinstance(dataframes, pd.DataFrame):
#         dataframes = [dataframes]
#     if isinstance(columns, str):
#         columns = [columns]

#     if columns is None:
#         raise ValueError("No columns provided.")

#     # use tqdm to show progress bar
#     tqdm.pandas()

#     result = []
#     for df in tqdm(dataframes):
#         df["arg_sentences"] = df["argument"].apply(lambda x: nltk.sent_tokenize(x))

#         # compute sentiment for each sentence in argument as a
#         # measure of the argumentativeness nature of the sentence
#         if sentiment:
#             sentiment_pipeline = pipeline(
#                 "sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english"
#             )
#             df["arg_sent_sentiment"] = df["arg_sentences"].progress_apply(
#                 lambda x: [sentiment_pipeline(s)[0]["score"] for s in x]
#             )
#             df["arg_sent_sentiment_avg"] = df["arg_sent_sentiment"].apply(
#                 lambda x: np.mean(x)
#             )

#         for col in columns:
#             new_word_col = col + "_word_list"
#             df[new_word_col] = df[col].apply(lambda x: nltk.word_tokenize(x))

#             # compute embeddings
#             if model:
#                 all_texts = df[col].tolist()
#                 all_embeddings = model.encode(all_texts)

#                 # assign the embeddings back to the rows
#                 new_emb_col = col + "_sent_emb"
#                 df[new_emb_col] = list(all_embeddings)

#                 # compute PCA for embeddings and keep x% of variance
#                 if use_pca:
#                     pca = PCA(n_components=variance, random_state=random_state)
#                     reduced_embeddings = pca.fit_transform(all_embeddings)

#                     new_pca_col = col + "_pca_emb"
#                     df[new_pca_col] = list(reduced_embeddings)

#         result.append(df)
#     return result


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


def run_evaluation(true_path: str, pred_path: str) -> str:
    """
    Build evaluation command for given script to compare true and predicted values.

    Args:
        true_path (str): Path to the file containing the true values.
        pred_path (str): Path to the file containing my predicted values.

    Returns:
        str: Command to run in terminal to evaluate the predictions.
    """
    return f"python eval.py --true {true_path} --predictions {pred_path}"
