"""Numpy Operations"""

import warnings
from io import TextIOWrapper
from typing import Any, List, Optional, Sequence

import numpy as np
from scipy.spatial.distance import jaccard, hamming
from scipy.stats import wasserstein_distance


def indices_to_binary(indices, length: int):
    result = np.zeros(length, dtype=int)
    result[indices] = 1
    return result


def to_1d(arrays: Sequence[Any]) -> Sequence[Any]:
    return [arr.reshape((-1,)) for arr in arrays]


def multi_label_binary(arr, threshold=0.5):
    return np.where(arr >= threshold, 1, 0)


def pad_pad(A, size):
    t = size - A.shape[0]
    if len(A.shape) == 2:
        pads = ((0, t), (0, 0))
    else:
        pads = (0, t)
    result = np.pad(A, pad_width=pads, mode="constant")
    # del A
    return result


def pad(A, size):
    # t = size - A.shape[0]
    if len(A.shape) == 2:
        """scikit-learn will convert all data to np.float32 beforehand.

        So we create arrays in that type to avoid copying.
        """
        result = np.zeros((size, A.shape[1]), dtype=np.float32)
        result[: A.shape[0], :] = A
    else:
        result = np.zeros((size,), dtype=np.float32)
        result[: A.shape[0]] = A
    return result


def mask(seq_length: int, pad_to_length: int):
    if pad_to_length < seq_length:
        raise ValueError("Pad length must greater or equal to sequence length")

    result = np.full((pad_to_length), True)
    result[:seq_length] = False

    return result


def cut(arr, lengths, padded):
    return [
        list(x[: lengths[i]]) + ([0] * (padded - lengths[i])) for i, x in enumerate(arr)
    ]

def compute_all_metrics_v2(
    exp_name: str, true, predict, lengths, file: Optional[TextIOWrapper] = None
):
    cut_true = cut(true, lengths, len(true[0]) - 1)
    cut_predict = cut(predict, lengths, len(predict[0]) - 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jaccard_result = [jaccard(cut_true[i], cut_predict[i]) for i in range(len(cut_true))]
        acc = 1 - np.mean(jaccard_result)
        print(f"[{exp_name}] jaccard: {acc}", file=file)
        
        hamming_result = [
            hamming(cut_true[i], cut_predict[i]) * len(cut_true[i])
            for i in range(len(cut_true))
        ]
        acc = np.mean(hamming_result)
        print(f"[{exp_name}] hamming: {acc}", file=file)
        
        wasserstein_result = [
            wasserstein_distance(cut_true[i], cut_predict[i]) * len(cut_true[i])
            for i in range(len(cut_true))
        ]
        acc = np.mean(wasserstein_result)
        print(f"[{exp_name}] wasserstein: {acc}", file=file)

    return jaccard_result, hamming_result, wasserstein_result


def filter_outliers(
    arrays: List[np.ndarray],
    max_value: Optional[int] = None,
    idx_filtered: Optional[List[int]] = None,
    percentile=99,
):
    if not idx_filtered:
        lengths = [arr.shape[0] for arr in arrays]
        lengths = np.array(lengths)
        if not max_value:
            max_value = int(np.percentile(lengths, percentile))
            print(f"max value: {max_value}")
        idx_filtered = [i for i, v in enumerate(lengths) if v <= max_value]

    arrays_filtered = [arrays[i] for i in idx_filtered]
    return arrays_filtered, idx_filtered, max_value


def filter_less_than_n_segment(
    arrays: List[np.ndarray],
    n: Optional[int] = 1,
    idx_filtered: Optional[List[int]] = None,
):
    if not idx_filtered:
        n_segments = [np.count_nonzero(arr) for arr in arrays]
        idx_filtered = [i for i, ns in enumerate(n_segments) if ns > n]
        # print(n_segments)

    arrays_filtered = [arrays[i] for i in idx_filtered]
    return arrays_filtered, idx_filtered


def gen_cumu_seq_1d(v: np.ndarray, pad_to: Optional[int] = None):
    nonzero_indices = np.nonzero(v)[0]
    # print(nonzero_indices)

    if pad_to:
        result = np.zeros((pad_to,))
    else:
        end_index = nonzero_indices[-1] + 1
        result = np.zeros((end_index,))

    if len(nonzero_indices) == 0:
        return result

    if len(nonzero_indices) == 1:
        result[:] = 1 / result.shape[0]
        return result

    curr_lst = [-1, *nonzero_indices[:-1]]
    next_lst = nonzero_indices

    for p, q in zip(curr_lst, next_lst):
        result[p + 1 : q + 1] = 1 / (q - p)

    return result


def gen_exp_cumu_seq_1d(v: np.ndarray, pad_to: Optional[int] = None):
    nonzero_indices = np.nonzero(v)[0]
    # print(nonzero_indices)

    if pad_to:
        result = np.zeros((pad_to,))
    else:
        end_index = nonzero_indices[-1] + 1
        result = np.zeros((end_index,))

    if len(nonzero_indices) == 0:
        return result

    if len(nonzero_indices) == 1:
        result[:] = 1 / result.shape[0]
        return result

    curr_lst = [-1, *nonzero_indices[:-1]]
    next_lst = nonzero_indices

    lookup_table = {}
    for p, q in zip(curr_lst, next_lst):
        # result[p+1:q+1] = 1 / (q - p)
        n_bucket = q - p
        for i, v in enumerate(range(p + 1, q + 1)):
            end = n_bucket - i - 1
            start = n_bucket - i
            end_value = _get_from_lookup_table(
                lookup_table, (end, n_bucket), lambda: _cdf_exp(end, n_bucket, rate=2.0)
            )
            # print(start, end)
            if start == n_bucket:
                result[v] = 1.0 - end_value
                if result[v] < 1e-6:
                    result[v] = 0.0
            else:
                start_value = _get_from_lookup_table(
                    lookup_table, (start, n_bucket), lambda: _cdf_exp(start, n_bucket, rate=2.0)
                )
                result[v] = start_value - end_value
                if result[v] < 1e-6:
                    result[v] = 0.0

    return result


def _get_from_lookup_table(lookup_table: dict, key: any, compute_func):
    if key not in lookup_table:
        lookup_table[key] = compute_func()

    return lookup_table[key]


def _cdf_exp(index: int, nrange: int, rate: float = 1.0):
    # print(f"_inv_cdf_exp: index={index}, nrange={nrange}, rate={rate}")
    # if nrange == 1:
    #     return 1.0

    value = 1 - np.exp(-rate * index)

    return value

def sos_to_np_array(data, dim: int):
    result = np.zeros((len(data), dim))
    for i, d in enumerate(data):
        if len(d) < dim:
            result[i, : len(d)] = np.array(d)
        else:
            result[i, :] = np.array(d)[:dim]
    return result
