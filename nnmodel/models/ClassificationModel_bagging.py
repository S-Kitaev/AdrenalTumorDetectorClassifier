"""
Deprecated compatibility shim.

Use nnmodel.models.ClassificationModel (bagging ensemble + diagnosis) instead.
"""

from nnmodel.models.ClassificationModel import ClassificationModel
from nnmodel.models.Simple3DCNN import Simple3DCNN

__all__ = ["ClassificationModel", "Simple3DCNN"]
