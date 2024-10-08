from django.db import models
from enum import Enum

class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class AuthenticationMethod(Enum):
    ICP = "ICP"
    FGR = "FGR"
    PNLK = "PNLK"
    FMR = "FMR"
    DGMR = "DGMR"
