from .hnts_mrg import (
    HNTSMRGDataset,
    HNTSMRGSliceDataset,
    HNTSMRGSequenceDataset,
    build_dataloaders,
)
from .brats_gli import BraTSGLIDataset, BraTSGLISliceDataset

try:
    from .levir_cd import LEVIRCDDataset
except ImportError:
    LEVIRCDDataset = None  # type: ignore[assignment,misc]

try:
    from .s2looking import S2LookingDataset
except ImportError:
    S2LookingDataset = None  # type: ignore[assignment,misc]

try:
    from .second import SECONDDataset
except ImportError:
    SECONDDataset = None  # type: ignore[assignment,misc]

try:
    from .xbd import XBDDataset
except ImportError:
    XBDDataset = None  # type: ignore[assignment,misc]

try:
    from .oaizib_cm import OAIZIBDataset, OAIZIBSliceDataset
except ImportError:
    OAIZIBDataset = None  # type: ignore[assignment,misc]
    OAIZIBSliceDataset = None  # type: ignore[assignment,misc]

try:
    from .ms_segmentation import MSSegmentationDataset
except ImportError:
    MSSegmentationDataset = None  # type: ignore[assignment,misc]

__all__ = [
    "HNTSMRGDataset",
    "HNTSMRGSliceDataset",
    "HNTSMRGSequenceDataset",
    "BraTSGLIDataset",
    "BraTSGLISliceDataset",
    "LEVIRCDDataset",
    "S2LookingDataset",
    "SECONDDataset",
    "XBDDataset",
    "OAIZIBDataset",
    "OAIZIBSliceDataset",
    "MSSegmentationDataset",
    "build_dataloaders",
]
