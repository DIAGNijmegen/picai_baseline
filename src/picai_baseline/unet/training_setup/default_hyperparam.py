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


unet_hyperparam = {
    'batch_size': 8,
    'model_strides': [(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)],
    'model_features': [32, 64, 128, 256, 512, 1024]
}


def get_default_hyperparams(args):
    """Retrieve default hyperparameters for given neural network architecture"""

    # used for inference
    if isinstance(args, dict):
        if args['model_type'] == 'unet':
            args['model_strides'] = unet_hyperparam['model_strides']
            args['model_features'] = unet_hyperparam['model_features']
        args = type('_', (object,), args)

    # used at train-time
    else:
        if args.model_type == 'unet':
            args.batch_size = unet_hyperparam['batch_size']
            args.model_strides = unet_hyperparam['model_strides']
            args.model_features = unet_hyperparam['model_features']
    return args
