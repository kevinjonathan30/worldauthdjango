from django.db import models
from enum import Enum

class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class AuthenticationMethod(Enum):
    ICP = "Iterative Closest Point"
    FGR = "Fast Global Registration"
    PNLK = "PointNetLK"
    FMR = "Feature-metric Registration"
    DGMR = "DeepGMR"
