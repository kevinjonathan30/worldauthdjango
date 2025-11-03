from ninja import Router, File, UploadedFile, Header
from ninja.responses import Response
from ninja.errors import HttpError
from typing import Optional
from .schemas import (
    PointCloudList,
    AuthResult,
    SuccessResponse,
    ErrorResponse,
    AuthMethod,
)
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

@router.get("/", include_in_schema=False)
def index(request):
    log_debug("accessIndex", extra={"message": "GET OUT OF HERE."})
    return HttpResponse("GET OUT OF HERE.")

@router.get(
    "/get-point-cloud",
    response={200: PointCloudList, 401: ErrorResponse, 500: ErrorResponse},
)
def get_point_cloud(request, authorization: str = Header(..., alias="U-Authorization")):
    """List uploaded point cloud filenames for the authenticated user.

    The U-Authorization header is required and used as the user folder name.
    """
    try:
        user_folder = os.path.join(settings.MEDIA_ROOT, authorization)
        os.makedirs(user_folder, exist_ok=True)
        pcd_list = os.listdir(user_folder)
        return Response({"success": True, "data": pcd_list}, status=200)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log_error("getPointCloud", extra={"message": f"Failed to list point clouds: {e}", "traceback": tb})
        # Raise an HTTP 500 with standardized ErrorResponse body
        raise HttpError(500, {"error": "Failed to list point clouds.", "detail": str(e)})

@router.delete(
    "/delete-point-cloud",
    response={200: SuccessResponse, 401: ErrorResponse, 400: ErrorResponse, 500: ErrorResponse},
)
def delete_point_cloud(request, authorization: str = Header(..., alias="U-Authorization"), FileRequest: str = Header(...)):
    """Delete a specific point cloud file for the authorized user.

    Headers:
    - U-Authorization: user folder name (required)
    - FileRequest: filename to delete (required)
    """
    storage = OverwriteStorage()
    try:
        file_path = os.path.join(settings.MEDIA_ROOT, authorization, FileRequest)
        if os.path.exists(file_path):
            storage.delete(file_path)
        delete_parent_directory(authorization)
        return Response({"success": True}, status=200)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log_error("deletePointCloud", extra={"message": f"Failed to delete point cloud: {e}", "traceback": tb})
        raise HttpError(500, {"error": "Failed to delete Point Cloud.", "detail": str(e)})

@router.post(
    "/register",
    response={200: SuccessResponse, 401: ErrorResponse, 400: ErrorResponse, 500: ErrorResponse},
)
def register(request, authorization: str = Header(..., alias="U-Authorization"), file: UploadedFile = File(...)):
    """Upload/register a single point cloud file for the authorized user.

    Headers:
    - U-Authorization: user folder name (required)

    Form-data:
    - file: .ply file to upload
    """
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S") + ".ply"
    try:
        file_path = os.path.join(settings.MEDIA_ROOT, authorization, dt_string)
        # read once into memory to avoid consuming the stream twice
        data = file.read()
        write_point_cloud(file_path, ContentFile(data))
        return Response({"success": True}, status=200)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log_error("registerPointCloud", extra={"message": f"Failed to register point cloud: {e}", "traceback": tb})
        raise HttpError(500, {"error": "Failed to register Point Cloud.", "detail": str(e)})

@router.post(
    "/authenticate",
    response={200: AuthResult, 401: ErrorResponse, 400: ErrorResponse, 500: ErrorResponse},
)
def authenticate(
    request,
    authorization: str = Header(..., alias="U-Authorization"),
    authMethod: Optional[AuthMethod] = Header(None, alias="AuthorizationMethod"),
    file: UploadedFile = File(...),
):
    """Authenticate uploaded file against all registered point clouds for the user.

    Headers:
    - U-Authorization: user folder name (required)
    - AuthorizationMethod: optional (ICP, FGR, PNLK, FMR, DGMR)

    Form-data:
    - file: .ply file to authenticate
    """
    if not authorization:
        raise HttpError(401, {"error": "Unauthorized", "detail": None})
    if not authMethod:
        authMethod = AuthMethod.ICP

    start = time()
    storage = OverwriteStorage()
    sourcePath = os.path.join(settings.MEDIA_ROOT, authorization, "source.ply")
    threshold = 0.7
    try:
        # read the uploaded file once
        data = file.read()
        write_point_cloud(sourcePath, ContentFile(data))
        pcd_source = o3d.io.read_point_cloud(sourcePath)
        highest_result = 0
        for file_name in os.listdir(os.path.join(settings.MEDIA_ROOT, authorization)):
            if file_name == "source.ply" or file_name == "uploaded.ply":
                continue
            target = os.path.join(settings.MEDIA_ROOT, authorization, file_name)
            pcd_target = o3d.io.read_point_cloud(target)
            if authMethod == AuthMethod.ICP:
                threshold = 0.35
                result = icp.execute_icp(pcd_source, pcd_target, threshold=threshold)
            elif authMethod == AuthMethod.FGR:
                threshold = 0.32
                result = icp.execute_icp(pcd_source, pcd_target, fast_registration=True, threshold=threshold)
            elif authMethod == AuthMethod.PNLK:
                threshold = 0.345
                result = pointnetlk.execute_pointnetlk(pcd_source, pcd_target, threshold=threshold)
            elif authMethod == AuthMethod.FMR:
                threshold = 0.35
                result = fmr.execute_fmr(pcd_source, pcd_target, threshold=threshold)
            elif authMethod == AuthMethod.DGMR:
                threshold = 0.47
                result = deepgmr.execute_deepgmr(pcd_source, pcd_target, threshold=threshold)
            highest_result = max(highest_result, result)
            if highest_result == 1:
                break
        authentication_time = time() - start
        log_info(
            "authenticatePointCloud",
            extra={
                "method": str(authMethod),
                "fitness": highest_result,
                "threshold": threshold,
                "authTime": authentication_time,
            },
        )
        return {"success": True, "fitness": highest_result}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log_error("authenticatePointCloud", extra={"message": f"Authentication failed: {e}", "traceback": tb})
        raise HttpError(500, {"error": "Authentication failed.", "detail": str(e)})
    finally:
        try:
            storage.delete(sourcePath)
        except Exception:
            pass
        delete_parent_directory(authorization)

@router.post(
    "/authenticate-two-point-clouds",
    response={200: AuthResult, 401: ErrorResponse, 400: ErrorResponse, 500: ErrorResponse},
)
def authenticate_two_point_clouds(
    request,
    authorization: str = Header(..., alias="U-Authorization"),
    authMethod: Optional[AuthMethod] = Header(None, alias="AuthorizationMethod"),
    file1: UploadedFile = File(...),
    file2: UploadedFile = File(...),
):
    """Authenticate using two uploaded point cloud files.

    Headers:
    - U-Authorization: user folder name (required)
    - AuthorizationMethod: optional, one of ICP, FGR, PNLK, FMR, DGMR

    Form-data:
    - file1: first .ply file
    - file2: second .ply file
    """
    if not authorization:
        raise HttpError(401, {"error": "Unauthorized", "detail": None})
    if not authMethod:
        authMethod = "ICP"

    write_start = time()
    storage = OverwriteStorage()
    sourcePath = os.path.join(settings.MEDIA_ROOT, authorization, "sourceFile.ply")
    targetPath = os.path.join(settings.MEDIA_ROOT, authorization, "targetFile.ply")

    # write uploaded files to disk
    write_point_cloud(sourcePath, ContentFile(file1.read()))
    write_point_cloud(targetPath, ContentFile(file2.read()))
    write_end = time()
    try:
        read_start = time()
        pcd_source = o3d.io.read_point_cloud(sourcePath)
        pcd_target = o3d.io.read_point_cloud(targetPath)
        read_end = time()
        auth_start = time()
        if authMethod == "ICP":
            threshold = 0.1
            fitness_boundary = 0.4
            result = icp.execute_icp(pcd_source, pcd_target, threshold=threshold)
        elif authMethod == "FGR":
            threshold = 0.18
            fitness_boundary = 0.4
            result = icp.execute_icp(pcd_source, pcd_target, fast_registration=True, threshold=threshold)
        elif authMethod == "PNLK":
            threshold = 0.37
            fitness_boundary = 0.4
            result = pointnetlk.execute_pointnetlk(pcd_source, pcd_target, threshold=threshold)
        elif authMethod == "FMR":
            threshold = 0.24
            fitness_boundary = 0.5
            result = fmr.execute_fmr(pcd_source, pcd_target, threshold=threshold)
        elif authMethod == "DGMR":
            threshold = 0.32
            fitness_boundary = 0.4
            result = deepgmr.execute_deepgmr(pcd_source, pcd_target, threshold=threshold)
        auth_end = time()
        log_info(
            "authenticatePointCloud",
            extra={
                "method": authMethod,
                "fitness": result,
                "threshold": threshold,
                "authTime": auth_end - auth_start,
                "readTime": read_end - read_start,
                "writeTime": write_end - write_start,
                "totalTime": (auth_end - write_start),
            }
        )
        return {"success": result >= fitness_boundary, "fitness": result}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log_error("authenticatePointCloud", extra={"message": f"Authentication failed: {e}", "traceback": tb})
        raise HttpError(500, {"error": "Authentication failed.", "detail": str(e)})
    finally:
        storage.delete(sourcePath)
        storage.delete(targetPath)
        delete_parent_directory(authorization)
