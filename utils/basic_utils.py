import numpy as np
import io
import os
import json
import logging
import random
import time
from collections import defaultdict, deque
import datetime
from pathlib import Path
from typing import List, Union

import torch
import torch.distributed as dist
from .distributed import is_dist_avail_and_initialized


logger = logging.getLogger(__name__)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window=100, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if meter.count == 0:  # skip empty meter
                loss_str.append(
                    "{}: {}".format(name, "No data")
                )
            else:
                loss_str.append(
                    "{}: {}".format(name, str(meter))
                )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            if meter.count == 0:
                loss_str.append(
                    "{}: {}".format(name, "No data")
                )
            else:
                loss_str.append(
                    "{}: {:.4f}".format(name, meter.global_avg)
                )
        return self.delimiter.join(loss_str)

    def get_global_avg_dict(self, prefix=""):
        """include a separator (e.g., `/`, or "_") at the end of `prefix`"""
        d = {f"{prefix}{k}": m.global_avg if m.count > 0 else 0. for k, m in self.meters.items()}
        return d

    def get_avg_dict(self, prefix=""):
        """include a separator (e.g., `/`, or "_") at the end of `prefix`"""
        d = {f"{prefix}{k}": m.avg if m.count > 0 else 0. for k, m in self.meters.items()}
        return d

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, log_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f} res mem: {res_mem:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % log_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB,
                        res_mem=torch.cuda.max_memory_reserved() / MB,
                    ))
                else:
                    logger.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def remove_files_if_exist(file_paths):
    for fp in file_paths:
        if os.path.isfile(fp):
            os.remove(fp)


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


def find_files_by_suffix_recursively(root: str, suffix: Union[str, List[str]]):
    """
    Args:
        root: path to the directory to start search files
        suffix: any str as suffix, or can match multiple such strings
            when input is List[str]. 
            Example 1, e.g., suffix: `.jpg` or [`.jpg`, `.png`]
            Example 2, e.g., use a `*` in the `suffix`: `START*.jpg.`.
    """
    if isinstance(suffix, str):
        suffix = [suffix, ]
    filepaths = flat_list_of_lists(
        [list(Path(root).rglob(f"*{e}")) for e in suffix])
    return filepaths


def match_key_and_shape(state_dict1, state_dict2):
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    print(f"keys1 - keys2: {keys1 - keys2}")
    print(f"keys2 - keys1: {keys2 - keys1}")

    mismatch = 0
    for k in list(keys1):
        if state_dict1[k].shape != state_dict2[k].shape:
            print(
                f"k={k}, state_dict1[k].shape={state_dict1[k].shape}, state_dict2[k].shape={state_dict2[k].shape}")
            mismatch += 1
    print(f"mismatch {mismatch}")


def merge_dicts(list_dicts):
    merged_dict = list_dicts[0].copy()
    for i in range(1, len(list_dicts)):
        merged_dict.update(list_dicts[i])
    return merged_dict


class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) for model parameters.
    
    Maintains an exponential moving average of model parameters, which often leads to
    better generalization and more stable models. The update formula is:
        ema_param = decay * ema_param + (1 - decay) * current_param
    
    Args:
        model: The model whose parameters will be averaged (can be DDP-wrapped)
        decay: The decay factor (default: 0.9999). Higher values make EMA change slower.
        device: Device to store EMA parameters on
        update_after_step: Only start updating EMA after this many steps (default: 0)
    """
    def __init__(self, model, decay=0.9999, device=None, update_after_step=0):
        self.decay = decay
        self.update_after_step = update_after_step
        self.num_updates = 0
        
        # Store reference to original model (may be DDP-wrapped)
        self.original_model = model
        
        # Get unwrapped model for parameter access
        if hasattr(model, 'module'):
            # DDP wrapped model
            self.model = model.module
        else:
            self.model = model
        
        # Initialize EMA state dict with model parameters
        self.ema_state_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                target_device = device if device is not None else param.data.device
                self.ema_state_dict[name] = param.data.clone().to(target_device)
    
    def update(self):
        """Update EMA parameters with current model parameters."""
        self.num_updates += 1
        
        # Skip update if not enough steps
        if self.num_updates < self.update_after_step:
            return
        
        # Compute decay factor (can adjust based on step number for warmup)
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        # Update EMA parameters - access from unwrapped model
        # This works because DDP models share parameters with their underlying module
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.ema_state_dict:
                if self.ema_state_dict[name].device != param.data.device:
                    self.ema_state_dict[name] = self.ema_state_dict[name].to(param.data.device)
                self.ema_state_dict[name].mul_(decay).add_(param.data, alpha=1 - decay)
    
    def state_dict(self):
        """Return the EMA state dict."""
        return {
            "ema_state_dict": {k: v.clone() for k, v in self.ema_state_dict.items()},
            "num_updates": self.num_updates
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state dict."""
        if isinstance(state_dict, dict) and "ema_state_dict" in state_dict:
            ema_state = state_dict["ema_state_dict"]
            self.num_updates = state_dict.get("num_updates", self.num_updates)
        else:
            ema_state = state_dict

        self.ema_state_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in ema_state:
                self.ema_state_dict[name] = ema_state[name].clone().to(param.data.device)
    
    def apply_to_model(self):
        """Apply EMA parameters to the model (for evaluation)."""
        # Save original parameters
        original_state_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                original_state_dict[name] = param.data.clone()
        
        # Apply EMA parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.ema_state_dict:
                param.data.copy_(self.ema_state_dict[name])
        
        return original_state_dict
    
    def restore_from_state_dict(self, original_state_dict):
        """Restore original model parameters after EMA evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in original_state_dict:
                param.data.copy_(original_state_dict[name])
