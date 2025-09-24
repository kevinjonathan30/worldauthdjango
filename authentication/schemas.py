from ninja import Schema, Field
from typing import List, Optional
from enum import Enum


class AuthMethod(str, Enum):
    ICP = "ICP"
    FGR = "FGR"
    PNLK = "PNLK"
    FMR = "FMR"
    DGMR = "DGMR"


class ErrorResponse(Schema):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information (for debugging)")


class SuccessResponse(Schema):
    success: bool = Field(..., description="Whether the operation succeeded")


class PointCloudList(Schema):
    success: bool
    data: List[str] = Field(..., description="List of point cloud filenames")


class AuthResult(Schema):
    success: bool
    fitness: Optional[float] = Field(None, description="Authentication fitness score (0-1)")
