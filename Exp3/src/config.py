# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
config settings, will be used in finetune.py
"""
from easydict import EasyDict as edict
import mindspore.common.dtype as mstype

optimizer_cfg = edict({
    'AdamWeightDecay': edict({
        'learning_rate': 3e-5,
        'end_learning_rate': 0.0,
        'power': 5.0,
        'weight_decay': 1e-5,
        'decay_filter': lambda x: 'layernorm' not in x.name.lower() and 'bias' not in x.name.lower(),
        'eps': 1e-6,
        'warmup_steps': 10000,
    }),
    'Lamb': edict({
        'learning_rate': 2e-5,
        'end_learning_rate': 0.0,
        'power': 1.0,
        'warmup_steps': 10000,
        'weight_decay': 0.01,
        'decay_filter': lambda x: 'layernorm' not in x.name.lower() and 'bias' not in x.name.lower(),
        'eps': 1e-6,
    }),
    'Momentum': edict({
        'learning_rate': 2e-5,
        'momentum': 0.9,
    }),
})
