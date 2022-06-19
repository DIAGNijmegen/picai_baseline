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

from picai_baseline.unet.training_setup.neural_networks.unets import UNet


def neural_network_for_run(args, device):
    """Select neural network architecture for given run"""

    if args.model_type == 'unet':
        model = UNet(
            spatial_dims=len(args.image_shape),
            in_channels=args.num_channels,
            out_channels=args.num_classes,
            strides=args.model_strides,
            channels=args.model_features
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    model = model.to(device)
    print("Loaded Neural Network Arch.:", args.model_type)
    return model
