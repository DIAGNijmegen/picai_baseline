#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import os
import torch


def compute_spec_for_run(args, train_device='cuda'):
    """Register GPU/CPU specifications"""
    device = torch.device(train_device)
    
    # os.sched_getaffinity(0) is not supported by all operating systems
    try:
        args.num_threads = np.min([len(os.sched_getaffinity(0)), args.max_threads])
    except:
        args.num_threads = args.max_threads
    
    print("-"*100)
    print("GPUs:", torch.cuda.get_device_name(0))
    print("CPU Threads:", args.num_threads)
    return device, args
