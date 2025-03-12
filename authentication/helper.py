import os

import open3d as o3d
import numpy as np
import requests

import google.auth
from google.auth.transport.requests import Request
import google.oauth2.service_account

from django.conf import settings

from .models import AuthenticationMethod
from logger import log_error

def write_point_cloud(filename, fileContent, format='ply', write_ascii=False, compressed=True):
    points_np = np.loadtxt(fileContent)
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(points_np)

    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    o3d.io.write_point_cloud(filename=filename, pointcloud=ply, format=format, write_ascii=write_ascii, compressed=compressed)

def get_method_config():
    credentials = google.oauth2.service_account.Credentials.from_service_account_file(
        os.path.join(settings.BASE_DIR, "config.json"),
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    credentials.refresh(Request())
    token = credentials.token

    url = 'https://firebaseremoteconfig.googleapis.com/v1/projects/{project_id}/remoteConfig'.format(
        project_id='worldauth-baec5' 
    )
    headers = {
        'Authorization': 'Bearer ' + token,
        'Content-Type': 'application/json',
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        config_data = response.json()
        key = "authenticationMethod"

        if 'parameters' in config_data and key in config_data['parameters']:
            return AuthenticationMethod(config_data['parameters'][key]['defaultValue']['value'])
        else:
            return AuthenticationMethod.ICP
    else:
        log_error(
            "getMethodConfig",
            extra={
                "message": f"Error: {response.status_code}, {response.text}"
            }
        )
        return AuthenticationMethod.ICP
