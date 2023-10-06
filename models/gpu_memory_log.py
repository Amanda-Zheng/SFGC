#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import gc
import datetime
import pynvml

import torch
import numpy as np
import sys

def _get_tensors():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            tensor = obj
        else:
            continue
        if tensor.is_cuda:
            yield tensor

def _write_log(f, write_str):
    print(write_str)
    f.write("%s\n" % write_str)

def gpu_memory_log(gpu_log_file="gpu_mem.log", device=0):
    stack_layer = 1
    func_name = sys._getframe(stack_layer).f_code.co_name
    file_name = sys._getframe(stack_layer).f_code.co_filename
    line = sys._getframe(stack_layer).f_lineno
    now_time = datetime.datetime.now()
    log_format = 'LINE:%s, FUNC:%s, FILE:%s, TIME:%s, CONTENT:%s'

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

    with open(gpu_log_file, 'a+') as f:
        write_str = log_format % (line, func_name, file_name, now_time, "")
        _write_log(f, write_str)

        ts_list = [tensor.size() for tensor in _get_tensors()]
        new_tensor_sizes = {(type(x),
                             tuple(x.size()),
                             ts_list.count(x.size()),
                             np.prod(np.array(x.size()))*4/1024**2)
                             for x in _get_tensors()}

        list_tensor_sizes = [(m*n, n, s, t) for t, s, n, m in new_tensor_sizes]
        list_tensor_sizes.sort(key=lambda x: x[0], reverse=True)
        for m, n, s, t in list_tensor_sizes:
            write_str = '[Memory: %s M | tensor: %s * Size:%s | %s]' %(str(m)[:5].ljust(5, '0'), str(n), str(s), str(t))
            _write_log(f, write_str)

        write_str = "memory_allocated:%f Mb" % float(torch.cuda.memory_allocated()/1024**2)
        _write_log(f, write_str)
        write_str = "max_memory_allocated:%f Mb" % float(torch.cuda.max_memory_allocated()/1024**2)
        _write_log(f, write_str)
        write_str = "memory_reserved:%f Mb" % float(torch.cuda.memory_reserved()/1024**2)
        _write_log(f, write_str)
        write_str = "max_memory_reserved:%f Mb" % float(torch.cuda.max_memory_reserved()/1024**2)
        _write_log(f, write_str)
        write_str = "Used Memory:%f Mb" % float(meminfo.used/1024**2)
        _write_log(f, write_str)
        write_str = "Free Memory:%f Mb" % float(meminfo.free/1024**2)
        _write_log(f, write_str)
        write_str = "Total Memory:%f Mb" % float(meminfo.total/1024**2)
        _write_log(f, write_str)

    pynvml.nvmlShutdown()
