from typing import Dict, List, Union

import nltk
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from tqdm import tqdm

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
    model: SentenceTransformer = None,
    use_pca: bool = True,
    variance: float = 0.95,
    random_state: int = 0,
) -> List[pd.DataFrame]:
    """
    Adds additional features to each DataFrame in the given list.

    Args:
        dataframes (List[pd.DataFrame]): A list of DataFrames.

    Returns:
        List[pd.DataFrame]: A list of modified DataFrames where the following features
        are added to each DataFrame:
        - 'sentences': Contains a list of sentences from the 'argument' column.
        - 'word_list': Contains a list of words from the 'argument' column.
        - 'sent_embeddings': Contains the sentence embeddings computed using the provided BERT model.
        - 'pca_sent_embeddings': Contains the PCA-transformed sentence embeddings, retaining 95% of variance.
    """

    # handle single df
    if isinstance(dataframes, pd.DataFrame):
        dataframes = [dataframes]

    # use tqdm to show progress bar
    tqdm.pandas()

    result = []
    for df in tqdm(dataframes):
        # split in sentences and words
        df["sentences"] = df["argument"].apply(lambda x: nltk.sent_tokenize(x))
        df["word_list"] = df["argument"].apply(lambda x: nltk.word_tokenize(x))

        # compute embeddings
        if model:
            # compute all embeddings at once
            all_sentences = [
                sentence for sentences in df["sentences"] for sentence in sentences
            ]
            sentence_indices = [
                index
                for index, sentences in enumerate(df["sentences"])
                for sentence in sentences
            ]
            all_embeddings = model.encode(all_sentences)

            # distribute the embeddings back to the rows
            df["sent_embeddings"] = [[] for _ in range(len(df))]
            for index, embedding in zip(sentence_indices, all_embeddings):
                df.at[index, "sent_embeddings"].append(embedding)

            # compute PCA for embeddings and keep x% of variance
            if use_pca:
                # fit on all embeddings of the given df
                pca = PCA(n_components=variance, random_state=random_state)
                pca.fit(all_embeddings)

                df["pca_sent_embeddings"] = df["sent_embeddings"].progress_apply(
                    lambda x: pca.transform(x)
                )

        result.append(df)
    return result


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
