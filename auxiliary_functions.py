from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
import multiprocessing as mp
import unittest

def hierarchical_fuzzy_dup_one_key(dup_df, key, short_dup_col, long_dup_col,
                                  scorer=fuzz.token_set_ratio, threshold=80,
                                  limit=50, look_for_dup_only_after_key=True):
    """
    Get the fuzzy duplicate associated to one row of a dataframe using
    a hierarchical approach to speed up the search.
    First we use the short_dup_col values to identify a short list of
    good candidates: making a comparison between the row *key* and all others
    rows of the dataframe based on *short_dup_col* column.
    Then we use the strings in *long_dup_col* columns of the short list to
    compare with the row *key* and identify final duplicated.

    :param dup_df: pandas dataframe used to look for duplicates.
    :param key: the index of the element for which we search duplicates.
    :param short_dup_col: str, name of the column used for shortlisting
    the candidates. The column should be a character column
    typically with shorter strings than in the long_dup_col column.
    :param long_dup_col: str, name of the column used for final search
    of duplicates. The column should be a character column
    typically with larger strings than in the short_dup_col column
    :param scorer: function used for scoring similarity between strings
    (the bigger the most similar). Typically those are ratios function
    from the package fuzzywuzzy.
    :param threshold: Decision threshold to decide if a candidate is a
    duplicate based on its score.
    :param limit: The maximal number of duplicate we are looking for.
    :param look_for_dup_only_after_key: If true, the duplicates are looked
    for only among the element of dup_df which index log is greater than
    the index loc of the element *key*. This is used typcally when the
    herarchical_fuzzy_dup_one_key function is applied along all element
    of dup_df to avoid repeating comparison.
    :return: tuple of 5 elements
    (key, the string we want to find, list of duplicates index,
    list of duplication score, list of duplicated string)
    """
    item = dup_df.loc[key, short_dup_col] # based string for the match search
    if look_for_dup_only_after_key:
        # filter only the part of dup_df located after the key.
        item_iloc = dup_df.index.get_loc(key)
        filtered_dup_df = dup_df.iloc[item_iloc:, ]
    else:
        filtered_dup_df = dup_df.copy()
    short_dupes = filtered_dup_df[short_dup_col].to_dict()
    short_dupes.pop(key) # remove item from the possible candidates.

    if len(short_dupes) > 0:
        # short list the matches using short_dup_col only.
        matches = process.extract(item, short_dupes, scorer=scorer, limit=limit)
        dup_candidates = [x for x in matches if x[1] > threshold]
        if len(dup_candidates) > 0:
            # look for the final matches using long_dup_col.
            dup_candidates_idx = [x[2] for x in dup_candidates]
            long_dupes = filtered_dup_df.loc[dup_candidates_idx, long_dup_col].\
                to_dict()
            item = dup_df.loc[key, long_dup_col]
            matches = process.extract(item, long_dupes, scorer=scorer,
                                      limit=limit)
            final_dup = [x for x in matches if x[1] > threshold]
            return (key, item, [x[2] for x in final_dup],
                    [x[1] for x in final_dup], [x[0] for x in final_dup])
        else:
            return (key, item, [], [], [])
    else:
        return (key, item, [], [], [])


def hierarchical_fuzzy_dup(dup_df, short_dup_col, long_dup_col,
                          scorer=fuzz.token_set_ratio, threshold=80, limit=50,
                          look_for_dup_only_after_key=True, n_proc=None):
    """
    Apply hierarchical_fuzzy_dup_one_key to each element of the dataframe dup_df
    The computation is parallelized on n_proc processors using the package
    multiprocessing.
    :param dup_df: pandas dataframe used to look for duplicates.
    :param short_dup_col: str, name of the column used for shortlisting
    the candidates. The column should be a character column
    typically with shorter strings than in the long_dup_col column.
    :param long_dup_col: str, name of the column used for final search
    of duplicates. The column should be a character column
    typically with larger strings than in the short_dup_col column
    :param scorer: function used for scoring similarity between strings
    (the bigger the most similar). Typically those are ratios function
    from the package fuzzywuzzy.
    :param threshold: Decision threshold to decide if a candidate is a
    duplicate based on its score.
    :param limit: The maximal number of duplicate we are looking for.
    :param look_for_dup_only_after_key: If true, the duplicates are looked
    for only among the element of dup_df which index log is greater than
    the index loc of the element *key*. This is used typcally to avoid
    repeating comparison.
    :param n_proc: Number of processor used to parallelize the task.
    :return: a data frame with five columns
        * source_index: index of df_dup
        * source_str: the reference string used for the duplicate search
        * dup_index: the list of index for found duplicates
        * dup_score: the list of score for found duplicates
        * dup_str: the list of string values for found duplicated
    """

    if n_proc is None:
        n_proc = mp.cpu_count()
    pool = mp.Pool(n_proc)
    # parallelize the match search.
    output = pool.starmap(
        hierarchical_fuzzy_dup_one_key,
        [(dup_df, key, short_dup_col, long_dup_col, scorer, threshold,
          limit, look_for_dup_only_after_key) for key in dup_df.index]
    )
    pool.close()
    output = pd.DataFrame(output,
                          columns=['source_index', 'source_str',
                                   'dup_index', 'dup_score', 'dup_str'])

    return output


def detect_duplicates(df, short_col_subset, long_col_subset,
                      scorer=fuzz.token_set_ratio, threshold=80, limit=50,
                      n_proc=None):
    """
    Detect fuzzy duplicates in a the dataframe df using a hierarchical search
    approach : preselect the duplicate candidate using the columns listed in
    short_col_subset, then make finale duplicates selection comparing
    the columns listed in short_col_subset. The output is a dataframe without
    fuzzy duplicates, the first occurrence of the duplicate is kept.
    :param df: pandas dataframe used to look for duplicates.
    :param short_col_subset: list, name of the columns used for first
    comparison so as to shortlist the duplicate candidates.
    :param long_col_subset: list, name of the columns used for the
    final search of duplicate. Typically long_col_subset will be bigger
    then short_col_subset and may contain all columns of the dataframe.
    :param scorer: function used for scoring similarity between strings
    (the bigger the most similar). Typically those are ratios function
    from the package fuzzywuzzy.
    :param threshold: Decision threshold to decide if a candidate is a
    duplicate based on its score.
    :param limit: The maximal number of duplicate we are looking for.
    :param n_proc: Number of processor used to parallelize the task.
    :return: tuple of two elements
    The dataframe without the duplicates. When a duplicate is found the first
    occurrence is kept.
    A data frame with five columns:
        * source_index: index of df_dup
        * source_str: the reference string used for the duplicate search
        * dup_index: the list of index for found duplicates
        * dup_score: the list of score for found duplicates
        * dup_str: the list of string values for found duplicated
    :return:
    """
    # fill na to be able to make comparison even when not all column are filled.
    dup_df = df.fillna('')
    # concatenate every elements in the short_col_subset into one
    # column which will be used for first string comparisons.
    dup_df['short_dup_col'] = dup_df[short_col_subset].\
        apply(lambda row: ' ; '.join(row.values.astype(str)), axis=1)
    dup_df.short_dup_col = dup_df.short_dup_col.str.lower()

    # concatenate every elements in the long_col_subset into one
    # column which will be used for second string comparisons.
    dup_df['long_dup_col'] = dup_df[long_col_subset].\
        apply(lambda row: ' ; '.join(row.values.astype(str)), axis=1)
    dup_df.long_dup_col = dup_df.long_dup_col.str.lower()

    # Get all the duplicated
    dup_candidates = hierarchical_fuzzy_dup(
        dup_df, 'short_dup_col', 'long_dup_col', scorer, threshold, limit,
        look_for_dup_only_after_key=True, n_proc=n_proc
    )

    # Drop duplicates. The indexes to be dropped will be all indexes
    # of the column "dup_index". Indeed for a group of duplicates
    # only the first to appear in the dataframe will not pe present
    # in this column (since we used the argument look_for_dup_only_after_key
    # when calling hierarchical_fuzzy_dup function.
    dup_to_drop_idx = np.concatenate(dup_candidates.dup_index.values)
    dup_to_drop_idx = np.unique(dup_to_drop_idx)
    dedupe_df = df.drop(dup_to_drop_idx, axis=0)

    return dedupe_df, dup_candidates



class TestDedupeFunctions(unittest.TestCase):

    def setUp(self):
        """Initialize tests."""
        dup_df = pd.DataFrame(
            {'firstname': ["joshua", "joshua", "joshua", "joshua", "clement"],
             'lastname': ["elrick", "elrick", "elrick", "elrack", "bizet"],
             'postcode': [2074, 2074, np.nan, 2075, 9212],
             'short_col': ["joshua; elrick;", "joshua; elrick;",
                           "joshua; elrick;", "joshua; elrack;",
                           "clement bizet"],
             'long_col': ["joshua; elrick; 2074", "joshua; elrick; 2074",
                          "joshua; elrick;", "joshua; elrack; 2074",
                          "clement; bizet; 9212"]})
        dup_df = dup_df.set_index(pd.Index([2, 1, 3, 5, 4]))
        self.dup_df = dup_df

    def test_hierarchical_fuzzy_dup_one_key_exact_dup(self):
        """Test that exact duplicates are seen by
        hierarchical_fuzzy_dup_one_key function"""
        key = 2
        short_dup_col = 'short_col'
        long_dup_col = 'long_col'
        output = hierarchical_fuzzy_dup_one_key(self.dup_df, key, short_dup_col,
                                                long_dup_col)
        self.assertIn(1, output[2])

    def test_hierarchical_fuzzy_dup_one_key_clear_not_dup(self):
        """Test that clear non duplicates are not in the results
        of hierarchical_fuzzy_dup_one_key function"""
        key = 2
        short_dup_col = 'short_col'
        long_dup_col = 'long_col'
        output = hierarchical_fuzzy_dup_one_key(self.dup_df, key, short_dup_col,
                                                long_dup_col)
        self.assertNotIn(4, output[2])

    def test_hierarchical_fuzzy_dup_one_key_after_key_true(self):
        """Test the parameter look_for_dup_only_after_key=True
        For key = 5, no duplicate should be found"""
        key = 5
        short_dup_col = 'short_col'
        long_dup_col = 'long_col'
        output = hierarchical_fuzzy_dup_one_key(self.dup_df, key, short_dup_col,
                                                long_dup_col)
        self.assertEqual(len(output[2]), 0)

    def test_hierarchical_fuzzy_dup_one_key_after_key_false(self):
        """Test the parameter look_for_dup_only_after_key=True
        For key = 5, 4 duplicates should be found"""
        key = 5
        short_dup_col = 'short_col'
        long_dup_col = 'long_col'
        output = hierarchical_fuzzy_dup_one_key(
            self.dup_df, key, short_dup_col, long_dup_col,
            look_for_dup_only_after_key=False
        )
        self.assertGreaterEqual(len(output[2]), 2)

    def test_detect_duplicates(self):
        """test detect duplicates function"""
        short_col = ['firstname', 'lastname']
        long_col = ['firstname', 'lastname', 'postcode']
        dedup_df, _ = detect_duplicates(self.dup_df, short_col,
                                        long_col, n_proc=1)
        self.assertListEqual(dedup_df.index.to_list(), [2, 4])


if __name__ == '__main__':
    run_test = True
    if run_test:
        unittest.main()