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

from typing import Any, Optional
try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass


def z_score_norm(image: "npt.NDArray[Any]", percentile: Optional[float] = None) -> "npt.NDArray[Any]":
    """
    Z-score normalization (mean=0; stdev=1), where intensities
    below or above the given percentile are discarded.
    [Ref: DLTK]

    Parameters:
    - image: N-dimensional image to be normalized
    - percentile: cutoff percentile (between 0-50)

    Returns:
    - scaled image
    """
    image = image.astype(np.float32)

    if percentile is not None:
        # clip distribution of intensity values
        lower_bnd = np.percentile(image, 100-percentile)
        upper_bnd = np.percentile(image, percentile)
        image = np.clip(image, lower_bnd, upper_bnd)

    # perform z-score normalization
    mean = np.mean(image)
    std = np.std(image)
    if std > 0:
        return (image - mean) / std
    else:
        return image * 0.
