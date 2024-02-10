from .backbone.main_backbone import mainBackboneLOCTSeg
from .segmentation_head import DeepLabHead
from .segmentation_model import _SMSimpleBackboneClassifier, _SMBackboneSkip

__all__ = [
    "BaseModel",
]


class baseModel(_SMSimpleBackboneClassifier):
    pass

def _BaseModel(
        backbone: mainBackboneLOCTSeg,
        num_classes: int,
) -> baseModel:
    classifier = DeepLabHead(128, num_classes)
    return baseModel(backbone, classifier)


def BaseModel(
        num_classes: int = 2,
        input_channels: int = 3
) -> baseModel:
    backbone = mainBackboneLOCTSeg(input_channels=input_channels)
    model = _BaseModel(backbone, num_classes)
    return model