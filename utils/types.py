from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

f64 = np.float64
i64 = np.int64
i32 = np.int32
NPFArray = npt.NDArray[f64]
NPIArray = npt.NDArray[i64]
NPBArray = npt.NDArray[np.bool_]
NPUIArray = npt.NDArray[np.uint8]
NPDTArray = npt.NDArray[np.datetime64]
XYLine = Tuple[NPFArray, NPFArray]


# from utils.types import NPFArray, NPIArray, NPBArray, XYLine, f64
# from utils.types import NPFArray, NPIArray, NPBArray, XYLine, f64
