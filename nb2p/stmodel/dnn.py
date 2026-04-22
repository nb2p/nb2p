import gc
import io
import os
from pprint import pprint
from random import sample
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union, cast
from regex import D
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from torch.nn import (
    LSTM,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from tqdm import tqdm

from nb2p import npop, codeop
from nb2p.stmodel import ModelInput
from nb2p.npop import multi_label_binary

RANDOM_SEED = 3407
N_ENCODING_DIM = 96
N_OUTPUT_DIM = 256


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers):
        super().__init__()

        encoder_layers = TransformerEncoderLayer(
            N_ENCODING_DIM, num_heads, dim_feedforward=hidden_size, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.linear1 = nn.Linear(96 * input_size, input_size)

    def forward(self, input_seq, src_mask, is_inference=False):
        if is_inference:
            enc_out = self.transformer_encoder(input_seq, src_key_padding_mask=None)
        else:
            enc_out = self.transformer_encoder(input_seq, src_key_padding_mask=src_mask)

        trans_out = enc_out.mean(dim=2, keepdim=True)
        trans_out = trans_out.reshape(enc_out.shape[0], -1)

        return trans_out


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.hidden_size = hidden_size

        self.lstm = LSTM(
            N_ENCODING_DIM * 2,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            proj_size=N_ENCODING_DIM // 2,
        )

    def forward(self, input_seq, src_mask, is_inference=False):
        input_seq = double_tensor_length(input_seq)
        trans_out = self.lstm(input_seq)[0]
        trans_out = trans_out.mean(dim=2)
        trans_out = trans_out.reshape(trans_out.shape[0], -1)

        return trans_out


def replace_elements(data, replacement_function):
    """
    Recursively replaces elements in an arbitrarily nested list based on a replacement function.

    Args:
        data: The input data, which can be a list or any other iterable.
        replacement_function: A function that takes an element and returns its replacement.

    Returns:
        A new list with replaced elements.
    """

    if not isinstance(data, list):
        return replacement_function(data)

    result = []
    for element in data:
        result.append(replace_elements(element, replacement_function))
    return result


def double_tensor_length(tensor: torch.Tensor):
    # use F.pad to double tensor length at the last dimension
    return F.pad(tensor, (0, tensor.shape[-1]))


class NB2PDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_internal_layers,
        epsilon_scale=1.0,
        device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
        setup="bce",
    ):
        super().__init__()
        self._epsilon_scale = epsilon_scale
        self._device = device

        self.hidden_size = hidden_size
        self._layer_width = N_ENCODING_DIM * 2

        self.inner_lstm = LSTM(
            self._layer_width,
            hidden_size,
            num_internal_layers,
            batch_first=True,
            bidirectional=True,
            proj_size=N_ENCODING_DIM // 2,
        )

        self._setup = setup

        self._dummy = torch.zeros(1, N_ENCODING_DIM).to(self._device)

        self._num_layers = num_layers

    def forward(self, input_batch, src_mask, is_inference=False):
        # a nested list.
        # first level: input
        # second level: AST children
        # third level: a tuple of (index expanded from the last layer, list of encodings)
        output_batch = []

        for inp in input_batch:
            # each sample
            decodings = {layer_i: [] for layer_i in range(len(inp))}

            def apply_lstm(i: int, cme: torch.Tensor, layer_i: int):
                output = self.inner_lstm(cme.unsqueeze(0))[0][0]
                output = output.mean(dim=0)
                decodings[layer_i - 1].append((i, output))
                # print(f"saved on upper layer [{layer_i - 1}]: index {i}")

            layer_iter_start = (
                len(inp) - 1 if len(inp) <= self._num_layers else self._num_layers - 1
            )
            for layer_i in range(layer_iter_start, 0, -1):
                layer = inp[layer_i]

                curr_layer_encodings = torch.cat([enc for _, enc in layer], dim=0)
                curr_layer_encodings = double_tensor_length(curr_layer_encodings)

                for curr_layer_i, dec in decodings[layer_i]:
                    curr_layer_encodings[curr_layer_i, N_ENCODING_DIM:] = dec

                start = -1
                value = -1
                count = 0
                for merge_i, (upper_layer_i, _) in enumerate(layer):
                    if start == -1:
                        start = merge_i
                        value = upper_layer_i

                    elif value != upper_layer_i:
                        # print("apply lstm on index range {}-{} with index {}".format(start, merge_i, upper_layer_i))
                        apply_lstm(
                            upper_layer_i,
                            curr_layer_encodings[start:merge_i, :],
                            layer_i,
                        )

                        start = merge_i
                        value = upper_layer_i
                        count += 1

                apply_lstm(value, curr_layer_encodings[start : len(layer), :], layer_i)

            layer = inp[0]
            curr_layer_encodings = []
            
            curr_layer_encodings = [enc for _, enc in layer]
            curr_layer_encodings += [
                self._dummy.expand(256 - len(layer), N_ENCODING_DIM)
            ]

            curr_layer_encodings = torch.cat(curr_layer_encodings, dim=0)
            curr_layer_encodings = double_tensor_length(curr_layer_encodings)

            for curr_layer_i, dec in decodings[0]:
                # print(f"get index {curr_layer_i} on layer [0]. shape: {dec.shape}")

                curr_layer_encodings[curr_layer_i, N_ENCODING_DIM:] = dec

            output_batch.append(curr_layer_encodings)

        result = torch.stack(output_batch, dim=0)
        result = self.inner_lstm(result)[0]
        result = result.mean(dim=2)
        result = result.reshape(len(output_batch), -1)

        return result


@dataclass
class DNNInput:
    X: ModelInput
    y: ModelInput
    code: str
    mask: ModelInput


@dataclass
class DNNInputBatches:
    X: Sequence[ModelInput]
    y: Sequence[ModelInput]
    code: Sequence[str]
    mask: Sequence[ModelInput]


criterion_bce = nn.BCEWithLogitsLoss()


def get_batch(data: ModelInput, batch_size: int):
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


METRIC_FUNC = npop.compute_all_metrics_v2


class Trainer:
    def __init__(
        self,
        model,
        num_epochs,
        batch_size,
        learning_rate,
        eval_freq,
        checkpoint_freq,
        device: torch.device,
        model_write_dir: Path,
        report_interval=100,
        model_name: Optional[str] = "nb2pss",
        window_size: Optional[int] = None,
        setup: str = "bce",
        start=0,
    ) -> None:
        self._model = model
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._eval_freq = eval_freq
        self._checkpoint_freq = checkpoint_freq
        self._device = device
        self._model_write_dir = model_write_dir
        self._report_interval = report_interval
        self._model_name = model_name
        self._setup = setup
        self._start = start
        if window_size is None:
            self._window_size = self._report_interval
        else:
            self._window_size = window_size

        self._build_setup(setup)

        if model_name == 'transformer':
            self._optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=0.01
            )
        else:
            self._optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=0.001
            )
        

    def _build_setup(self, setup: str):
        self._criterion = criterion_bce
        self._eval_func = self._do_evaluate
        self._pred_func = npop.multi_label_binary

    @property
    def _log_file_path(self):
        return self._model_write_dir / f"{self._model_name}.log"

    def _should_evaluate(self, epoch: int):
        return (epoch + 1) % self._eval_freq == 0

    def _should_checkpoint(self, epoch: int):
        return (epoch + 1) % self._checkpoint_freq == 0

    def _checkpoint(self, epoch: int):
        torch.save(
            self._model.state_dict(),
            self._model_write_dir / f"{self._model_name}-e{epoch + 1}.pt",
        )
        print("Model saved")

    def _log_evaluate(self, epoch: int, train_time, eval_loss, eval_acc, f):
        log_str = (
            "Epoch {}: "
            "Train Time = {:.4f}, "
            "Eval Loss = {:.4f}, "
            "Eval Accuracy = {:.4f}".format(epoch + 1, train_time, eval_loss, eval_acc)
        )
        print(log_str)
        print(log_str, file=f)

    def _should_report(self, i: int):
        return i % self._report_interval == (self._report_interval - 1)

    def train(
        self, train_data: DNNInput, test_data: DNNInput, n_try: Optional[int] = None
    ):
        X_train_batched = get_batch(train_data.X, self._batch_size)
        y_train_batched = get_batch(train_data.y, self._batch_size)
        train_mask_batched = get_batch(train_data.mask, self._batch_size)

        X_test_batched = get_batch(test_data.X, self._batch_size)
        y_test_batched = get_batch(test_data.y, self._batch_size)
        code_test_batched = get_batch(test_data.code, self._batch_size)
        test_mask_batched = get_batch(test_data.mask, self._batch_size)

        test_lengths = list(map(lambda x: (~x).sum(), test_data.mask))

        train_time = 0.0

        for epoch in range(0, self._start):
            X_train_batched, y_train_batched = cast(
                Tuple[ModelInput, ModelInput],
                shuffle(X_train_batched, y_train_batched, random_state=RANDOM_SEED),
            )

        for epoch in range(self._start, self._num_epochs):
            # torch.cuda.empty_cache()
            # gc.collect()

            X_train_batched, y_train_batched = cast(
                Tuple[ModelInput, ModelInput],
                shuffle(X_train_batched, y_train_batched, random_state=RANDOM_SEED),
            )
            # epoch_loss = 0.0
            loss_window = []

            progress_bar = tqdm(range(len(X_train_batched)), desc=f"Batch {epoch}")

            self._model.train()

            start_time = time.time()

            for i in progress_bar:
                if n_try and i >= n_try:
                    break

                if "dfgtree" in self._setup:
                    batch_X = X_train_batched[i]
                else:
                    batch_X = (
                        torch.tensor(np.array(X_train_batched[i], dtype=np.float64))
                        .float()
                        .to(self._device)
                    )

                batch_mask = torch.tensor(np.array(train_mask_batched[i])).to(
                    self._device
                )
                if len(batch_X) != len(batch_mask):
                    continue

                batch_y_np = np.array(y_train_batched[i], dtype=np.float64)
                batch_y = torch.tensor(batch_y_np).float().to(self._device)

                self._optimizer.zero_grad()

                y_pred = self._model(batch_X, batch_mask)

                if "bce" in self._setup:
                    batch_value = batch_y[~batch_mask]
                    n_neg = (batch_value == 0).sum()
                    n_pos = (batch_value == 1).sum()
                    if n_pos == 0:
                        pos_weight = torch.ones((1,)).to(self._device) * N_OUTPUT_DIM
                    else:
                        pos_weight = n_neg / n_pos
                    self._criterion = nn.BCEWithLogitsLoss(pos_weight=0.35 * pos_weight)

                    loss = self._criterion(y_pred[~batch_mask], batch_y[~batch_mask])
                else:
                    loss = self._criterion(y_pred, batch_y)
                    loss = loss[~batch_mask]

                if len(loss_window) >= self._window_size:
                    loss_window.pop(0)

                loss_window.append(loss.item())

                postfix_dict = {
                    "loss": loss.item(),
                    "aloss": sum(loss_window) / len(loss_window),
                }

                if self._should_report(i):
                    with open(self._log_file_path, "a") as f:
                        y_pred_0_cpu = y_pred[0].detach().cpu()
                        print(f"n_seg_pd: {y_pred_0_cpu}", file=f)
                        print(
                            f"n_seg_pd: {self._pred_func(y_pred_0_cpu, threshold=0.0)}",
                            file=f,
                        )
                        print(f"n_seg_gt: {self._pred_func(batch_y_np[0])}", file=f)
                        print(f"postfix_dict: {postfix_dict}", file=f)

                # if i % 10 == 0:
                progress_bar.set_postfix(postfix_dict)

                loss.backward()
                self._optimizer.step()

            train_time += time.time() - start_time

            if self._should_evaluate(epoch):
                self._model.eval()

                f = open(self._log_file_path, "a")

                eval_loss, eval_acc, _ = self._eval_func(
                    DNNInputBatches(
                        X=X_test_batched, y=y_test_batched, code=code_test_batched, mask=test_mask_batched
                    ),
                    n_try,
                    f"Epoch {epoch}",
                    f,
                    test_lengths,
                )
                eval_loss = eval_loss.mean().detach().cpu()
                eval_acc = eval_acc.item()

                self._log_evaluate(epoch, train_time, eval_loss, eval_acc, f)

                f.close()

                if self._should_checkpoint(epoch):
                    self._checkpoint(epoch)

            # self._scheduler.step()

    def _do_evaluate(
        self,
        batches: DNNInputBatches,
        n_try: Optional[int],
        exp_name: str,
        log_file: io.TextIOWrapper,
        test_lengths: list,
    ):
        eval_loss = 0.0

        code = []
        y_predicts = []
        y_gt_list = []
        y_predict_list = []
        for k in tqdm(range(len(batches.X)), total=len(batches.X)):
            if n_try and k >= n_try:
                break

            if "dfgtree" in self._setup:
                test_batch_X = batches.X[k]
            else:
                test_batch_X = torch.tensor(batches.X[k]).to(self._device)

            test_batch_y = torch.tensor(batches.y[k]).to(self._device)
            batch_mask = torch.tensor(np.array(batches.mask[k], dtype=np.float64)).to(
                self._device
            )
            y_pred_eval = self._model(test_batch_X, batch_mask, is_inference=True)
            y_pred = y_pred_eval.detach().cpu()
            y_predicts.append(y_pred)

            y_gt_list.extend(batches.y[k].tolist())
            y_predict_list.extend(y_pred.tolist())
            code.extend(batches.code[k])

            # print(len(y_gt_list), len(y_predict_list), len(code))

            y_pred_eval = y_pred_eval * batch_mask
            test_batch_y = test_batch_y * batch_mask

            eval_loss += self._criterion(y_pred_eval, test_batch_y).detach().cpu()

            del test_batch_X
            del test_batch_y
            del batch_mask
            del y_pred_eval

        # torch.cuda.empty_cache()
        # gc.collect()

        eval_loss /= len(batches.X)

        y_predict = np.concatenate(y_predicts)
        y_predict_cls = npop.multi_label_binary(y_predict)

        y_gt = np.concatenate(batches.y[: len(y_predicts)])

        print(y_gt.shape, y_predict_cls.shape)

        eval_acc = METRIC_FUNC(exp_name, y_gt, y_predict_cls, test_lengths)
        eval_acc = METRIC_FUNC(
            exp_name, y_gt, y_predict_cls, test_lengths, file=log_file
        )
        
        return eval_loss, eval_acc, [{
            "code": code[i], 
            "pred_ast": codeop.binary_to_segment_ends(y_predict_list[i]),
            "gt_ast": codeop.binary_to_segment_ends(y_gt_list[i]),
        } for i in range(len(code))]

    def evaluate(
        self,
        test_data: DNNInput,
        n_try: Optional[int] = None,
        epoch: int = 0,
    ):
        X_test_batched = get_batch(test_data.X, self._batch_size)
        y_test_batched = get_batch(test_data.y, self._batch_size)
        code_test_batched = get_batch(test_data.code, self._batch_size)
        test_mask_batched = get_batch(test_data.mask, self._batch_size)
        
        test_lengths = list(map(lambda x: (~x).sum(), test_data.mask))

        with torch.no_grad():
            f = open(self._log_file_path, "a")

            eval_loss, eval_acc, segment_result = self._eval_func(
                DNNInputBatches(
                    X=X_test_batched, y=y_test_batched, code=code_test_batched, mask=test_mask_batched
                ),
                n_try,
                f"Epoch {epoch}",
                f,
                test_lengths,
            )

            log_str = "Eval Loss = {:.4f}, Eval Accuracy = {:.4f}".format(
                eval_loss, eval_acc
            )
            print(log_str)
            print(log_str, file=f)

            f.close()

            # torch.cuda.empty_cache()
            # gc.collect()

        return segment_result



# Old training function (for TF)
def train(
    model,
    train_data: DNNInput,
    test_data: DNNInput,
    num_epochs,
    batch_size,
    learning_rate,
    eval_freq,
    checkpoint_freq,
    device: torch.device,
    model_write_dir: Path,
    n_try: Optional[int] = None,
    report_interval=100,
    model_name: Optional[str] = "nb2pss",
    start=0,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.001
    )

    X_train_batched = get_batch(train_data.X, batch_size)
    y_train_batched = get_batch(train_data.y, batch_size)
    train_mask_batched = get_batch(train_data.mask, batch_size)

    X_test_batched = get_batch(test_data.X, batch_size)
    y_test_batched = get_batch(test_data.y, batch_size)
    test_mask_batched = get_batch(test_data.mask, batch_size)

    test_lengths = list(map(lambda x: (~x).sum(), test_data.mask))

    train_time = 0.0

    for epoch in range(0, start):
        X_train_batched, y_train_batched = cast(
            Tuple[ModelInput, ModelInput],
            shuffle(X_train_batched, y_train_batched, random_state=RANDOM_SEED),
        )

    for epoch in range(start, num_epochs):
        torch.cuda.empty_cache()
        gc.collect()
        
        X_train_batched, y_train_batched = cast(
            Tuple[ModelInput, ModelInput],
            shuffle(X_train_batched, y_train_batched, random_state=RANDOM_SEED),
        )
        epoch_loss = 0.0

        progress_bar = tqdm(range(len(X_train_batched)), desc=f"Batch {epoch}")

        model.train()

        start_time = time.time()

        for i in progress_bar:
            if n_try and i >= n_try:
                break

            batch_X = (
                torch.tensor(np.array(X_train_batched[i], dtype=np.float64))
                .float()
                .to(device)
            )
            batch_mask = torch.tensor(np.array(train_mask_batched[i])).to(device)
            if batch_X.shape[0] != batch_mask.shape[0]:
                continue

            batch_y_np = np.array(y_train_batched[i], dtype=np.float64)
            batch_y = torch.tensor(batch_y_np).float().to(device)

            optimizer.zero_grad()

            y_pred = model(batch_X, batch_mask)
            
            batch_value = batch_y[~batch_mask]
            n_neg = (batch_value == 0).sum()
            n_pos = (batch_value == 1).sum()
            if n_pos == 0:
                pos_weight = torch.ones((1,)).to(device) * N_OUTPUT_DIM
            else:
                pos_weight = n_neg / n_pos
            criterion = nn.BCEWithLogitsLoss(pos_weight=0.35 * pos_weight)
            loss = criterion(y_pred[~batch_mask], batch_y[~batch_mask])

            progress_bar.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_X) / len(train_data.X)

        train_time += time.time() - start_time

        if (epoch + 1) % eval_freq == 0:
            model.eval()

            f = open(model_write_dir / f"{model_name}.log", "a")

            eval_loss, eval_acc = _do_evaluate(
                model,
                DNNInputBatches(
                    X=X_test_batched, y=y_test_batched, code=None, mask=test_mask_batched
                ),
                n_try,
                device,
                f"Epoch {epoch}",
                f,
                test_lengths,
            )

            log_str = (
                "Epoch {}: "
                "Train Time = {:.4f}, "
                "Train Loss = {:.4f}, "
                "Eval Loss = {:.4f}, "
                "Eval Accuracy = {:.4f}".format(
                    epoch + 1, train_time, epoch_loss, eval_loss, eval_acc
                )
            )
            print(log_str)
            print(log_str, file=f)

            f.close()

            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(
                    model.state_dict(),
                    model_write_dir / f"{model_name}-e{epoch + 1}.pt",
                )
                print("Model saved")


def _do_evaluate(
    model: nn.Module,
    batches: DNNInputBatches,
    n_try: Optional[int],
    device: torch.device,
    exp_name: str,
    log_file: io.TextIOWrapper,
    test_lengths: list,
):
    eval_loss = 0.0

    y_predicts = []
    for k in tqdm(range(len(batches.X)), total=len(batches.X)):
        if n_try and k >= n_try:
            break

        test_batch_X = torch.tensor(batches.X[k]).to(device)
        test_batch_y = torch.tensor(batches.y[k]).to(device)
        batch_mask = torch.tensor(batches.mask[k]).float().to(device)
        y_pred_eval = model(test_batch_X, batch_mask, is_inference=True)
        y_predicts.append(y_pred_eval.detach().cpu())

        y_pred_eval = y_pred_eval * batch_mask
        test_batch_y = test_batch_y * batch_mask

        del test_batch_X
        del test_batch_y
        del batch_mask
        del y_pred_eval

    y_predict = np.concatenate(y_predicts)
    y_predict_cls = npop.multi_label_binary(y_predict)

    y_gt = np.concatenate(batches.y[: len(y_predict)])

    eval_acc = npop.compute_all_metrics_v2(exp_name, y_gt, y_predict_cls, test_lengths)
    eval_acc = npop.compute_all_metrics_v2(
        exp_name, y_gt, y_predict_cls, test_lengths, file=log_file
    )

    return eval_loss, eval_acc


def evaluate(
    model: nn.Module,
    test_data: DNNInput,
    batch_size,
    device,
    model_write_dir: Path,
    n_try: Optional[int] = None,
    model_name: str = "nb2pss",
):
    X_test_batched = get_batch(test_data.X, batch_size)
    y_test_batched = get_batch(test_data.y, batch_size)
    test_mask_batched = get_batch(test_data.mask, batch_size)

    test_lengths = list(map(lambda x: (~x).sum(), test_data.mask))

    with torch.no_grad():
        f = open(model_write_dir / f"{model_name}.log", "a")

        eval_loss, eval_acc = _do_evaluate(
            model,
            DNNInputBatches(X=X_test_batched, y=y_test_batched, code=None, mask=test_mask_batched),
            n_try,
            device,
            model_name,
            f,
            test_lengths,
        )

        log_str = "Eval Loss = {:.4f}, Eval Accuracy = {:.4f}".format(
            eval_loss, eval_acc
        )
        print(log_str)
        print(log_str, file=f)

        f.close()