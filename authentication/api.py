from ninja import Router
from django.http import HttpResponse
from django.conf import settings
from .storage import OverwriteStorage
from authentication.icp import icp
from authentication.pointnetlk import pointnetlk
from authentication.fmr import fmr
from authentication.deepgmr import deepgmr
from authentication.helper import write_point_cloud, delete_parent_directory
import os
from datetime import datetime
from time import time
import open3d as o3d
from django.core.files.base import ContentFile
from logger import log_debug, log_info, log_error

router = Router()

@router.get("/")
def index(request):
    log_debug("accessIndex", extra={"message": "GET OUT OF HERE."})
    return HttpResponse("GET OUT OF HERE.")

@router.get("/get_point_cloud")
def get_point_cloud(request):
    authorization = request.headers.get('Authorization')
    if not authorization:
        return {"error": "Unauthorized"}
    try:
        user_folder = os.path.join(settings.MEDIA_ROOT, authorization)
        os.makedirs(user_folder, exist_ok=True)
        pcd_list = os.listdir(user_folder)
        return {"success": True, "data": pcd_list}
    except Exception as e:
        log_error("getPointCloud", extra={"message": f"Failed to list point clouds: {e}"})
        return {"success": True, "data": []}

@router.delete("/delete_point_cloud")
def delete_point_cloud(request):
    authorization = request.headers.get('Authorization')
    file_request = request.headers.get('FileRequest')
    if not authorization:
        return {"error": "Unauthorized"}
    if not file_request:
        return {"error": "Invalid/No filename"}
    storage = OverwriteStorage()
    try:
        file_path = os.path.join(settings.MEDIA_ROOT, authorization, file_request)
        if os.path.exists(file_path):
            storage.delete(file_path)
        delete_parent_directory(authorization)
        return {"success": True}
    except Exception as e:
        log_error("deletePointCloud", extra={"message": f"Failed to delete point cloud: {e}"})
        return {"success": False, "error": "Failed to delete Point Cloud."}

@router.post("/register")
def register(request):
    authorization = request.headers.get('Authorization')
    if not authorization:
        return {"error": "Unauthorized"}
    if 'files' not in request.FILES or len(request.FILES.getlist('files')) != 1:
        return {"error": "Invalid number of files were uploaded."}
    files = request.FILES.getlist('files')
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S") + ".ply"
    try:
        file_path = os.path.join(settings.MEDIA_ROOT, authorization, dt_string)
        write_point_cloud(file_path, ContentFile(files[0].read()))
        return {"success": True}
    except Exception as e:
        log_error("registerPointCloud", extra={"message": f"Failed to register point cloud: {e}"})
        return {"success": False, "error": "Failed to register Point Cloud."}

@router.post("/authenticate")
def authenticate(request):
    authorization = request.headers.get('Authorization')
    authMethod = request.headers.get('AuthorizationMethod')
    if not authorization:
        return {"error": "Unauthorized"}
    if 'files' not in request.FILES or len(request.FILES.getlist('files')) != 1:
        return {"error": "Invalid number of files were uploaded."}
    if not authMethod:
        authMethod = "ICP"
    start = time()
    files = request.FILES.getlist('files')
    storage = OverwriteStorage()
    sourcePath = os.path.join(settings.MEDIA_ROOT, authorization, "source.ply")
    threshold = 0.7
    try:
        write_point_cloud(sourcePath, ContentFile(files[0].read()))
        pcd_source = o3d.io.read_point_cloud(sourcePath)
        highest_result = 0
        for file in os.listdir(os.path.join(settings.MEDIA_ROOT, authorization)):
            if file == "source.ply" or file == "uploaded.ply":
                continue
            target = os.path.join(settings.MEDIA_ROOT, authorization, file)
            pcd_target = o3d.io.read_point_cloud(target)
            if authMethod == "ICP":
                threshold = 0.35
                result = icp.execute_icp(pcd_source, pcd_target, threshold=threshold)
            elif authMethod == "FGR":
                threshold = 0.32
                result = icp.execute_icp(pcd_source, pcd_target, fast_registration=True, threshold=threshold)
            elif authMethod == "PNLK":
                threshold = 0.345
                result = pointnetlk.execute_pointnetlk(pcd_source, pcd_target, threshold=threshold)
            elif authMethod == "FMR":
                threshold = 0.35
                result = fmr.execute_fmr(pcd_source, pcd_target, threshold=threshold)
            elif authMethod == "DGMR":
                threshold = 0.47
                result = deepgmr.execute_deepgmr(pcd_source, pcd_target, threshold=threshold)
            highest_result = max(highest_result, result)
            if highest_result == 1:
                break
        authentication_time = time() - start
        log_info(
            "authenticatePointCloud",
            extra={
                "method": authMethod,
                "fitness": highest_result,
                "threshold": threshold,
                "authTime": authentication_time
            }
        )
        return {"success": True, "fitness": highest_result}
    except Exception as e:
        log_error("authenticatePointCloud", extra={"message": f"Authentication failed: {e}"})
        return {"error": "No registered point cloud."}
    finally:
        storage.delete(sourcePath)
        delete_parent_directory(authorization)

@router.post("/authenticate_two")
def authenticate_two(request):
    authorization = request.headers.get('Authorization')
    authMethod = request.headers.get('AuthorizationMethod')
    if not authorization:
        return {"error": "Unauthorized"}
    if 'files' not in request.FILES or len(request.FILES.getlist('files')) != 2:
        return {"error": "Invalid number of files were uploaded."}
    if not authMethod:
        authMethod = "ICP"
    start = time()
    files = request.FILES.getlist('files')
    storage = OverwriteStorage()
    sourcePath = os.path.join(settings.MEDIA_ROOT, authorization, "sourceFile.ply")
    targetPath = os.path.join(settings.MEDIA_ROOT, authorization, "targetFile.ply")
    write_point_cloud(sourcePath, ContentFile(files[0].read()))
    write_point_cloud(targetPath, ContentFile(files[1].read()))
    try:
        pcd_source = o3d.io.read_point_cloud(sourcePath)
        pcd_target = o3d.io.read_point_cloud(targetPath)
        if authMethod == "ICP":
            threshold = 0.35
            result = icp.execute_icp(pcd_source, pcd_target, threshold=threshold)
        elif authMethod == "FGR":
            threshold = 0.32
            result = icp.execute_icp(pcd_source, pcd_target, fast_registration=True, threshold=threshold)
        elif authMethod == "PNLK":
            threshold = 0.345
            result = pointnetlk.execute_pointnetlk(pcd_source, pcd_target, threshold=threshold)
        elif authMethod == "FMR":
            threshold = 0.35
            result = fmr.execute_fmr(pcd_source, pcd_target, threshold=threshold)
        elif authMethod == "DGMR":
            threshold = 0.47
            result = deepgmr.execute_deepgmr(pcd_source, pcd_target, threshold=threshold)
        authentication_time = time() - start
        log_info(
            "authenticatePointCloud",
            extra={
                "method": authMethod,
                "fitness": result,
                "threshold": threshold,
                "authTime": authentication_time
            }
        )
        return {"success": True, "fitness": result}
    except Exception as e:
        log_error("authenticatePointCloud", extra={"message": f"Authentication failed: {e}"})
        return {"error": "No registered point cloud."}
    finally:
        storage.delete(sourcePath)
        storage.delete(targetPath)
        delete_parent_directory(authorization)
