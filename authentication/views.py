from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.conf import settings

from .storage import OverwriteStorage
from authentication.icp import icp
from authentication.pointnetlk import pointnetlk
from authentication.fmr import fmr
from authentication.deepgmr import deepgmr
from authentication.helper import write_point_cloud

import os
from datetime import datetime
from time import time
from logger import log_debug, log_info, log_error
import open3d as o3d
import tempfile

def index(request):
    log_debug("accessIndex", extra={"message": "GET OUT OF HERE."})
    return HttpResponse("GET OUT OF HERE.")

@csrf_exempt
def get_point_cloud(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Invalid request'}, status=400)
    authorization = request.headers.get('Authorization')
    if not authorization:
        return JsonResponse({'error': 'Unauthorized'}, status=401)
    try:
        pcd_list = os.listdir(os.path.join(settings.MEDIA_ROOT, authorization))
        return JsonResponse({'success': True, 'data': pcd_list}, status=200)
    except Exception as e:
        log_error("getPointCloud", extra={"message": f"Failed to list point clouds: {e}"})
        return JsonResponse({'success': True, 'data': []}, status=200)

@csrf_exempt
def delete_point_cloud(request):
    if request.method != 'DELETE':
        return JsonResponse({'error': 'Invalid request'}, status=400)
    authorization = request.headers.get('Authorization')
    file_request = request.headers.get('FileRequest')
    if not authorization:
        return JsonResponse({'error': 'Unauthorized'}, status=401)
    if not file_request:
        return JsonResponse({'error': 'Invalid/No filename'}, status=400)
    storage = OverwriteStorage()
    try:
        storage.delete(os.path.join(authorization, file_request))
        remaining_files = os.listdir(os.path.join(settings.MEDIA_ROOT, authorization))
        if not remaining_files:
            storage.delete(authorization)
        return JsonResponse({'success': True}, status=200)
    except Exception as e:
        log_error("deletePointCloud", extra={"message": f"Failed to delete point cloud: {e}"})
        return JsonResponse({'success': False, 'error': 'Failed to delete Point Cloud.'}, status=500)

@csrf_exempt
def register(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request'}, status=400)
    authorization = request.headers.get('Authorization')
    if not authorization:
        return JsonResponse({'error': 'Unauthorized'}, status=401)
    if 'files' not in request.FILES or len(request.FILES.getlist('files')) != 1:
        return JsonResponse({'error': 'Invalid number of files were uploaded.'}, status=400)
    files = request.FILES.getlist('files')
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S") + ".ply"
    try:
        file_path = os.path.join(settings.MEDIA_ROOT, authorization, dt_string)
        write_point_cloud(file_path, ContentFile(files[0].read()))
        return JsonResponse({'success': True}, status=200)
    except Exception as e:
        log_error("registerPointCloud", extra={"message": f"Failed to register point cloud: {e}"})
        return JsonResponse({'success': False, 'error': 'Failed to register Point Cloud.'}, status=500)

@csrf_exempt
def authenticate(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request'}, status=400)
    authorization = request.headers.get('Authorization')
    authMethod = request.headers.get('AuthorizationMethod')
    if not authorization:
        return JsonResponse({'error': 'Unauthorized'}, status=401)
    if 'files' not in request.FILES or len(request.FILES.getlist('files')) != 1:
        return JsonResponse({'error': 'Invalid number of files were uploaded.'}, status=400)
    if not authMethod:
        authMethod = "ICP"
    start = time()
    files = request.FILES.getlist('files')
    threshold = 0.7
    try:
        # Use tempfile to read uploaded file into Open3D point cloud (workaround for bytes issue)
        uploaded_bytes = files[0].read()
        with tempfile.NamedTemporaryFile(suffix=".ply") as tmp:
            tmp.write(uploaded_bytes)
            tmp.flush()
            pcd_source = o3d.io.read_point_cloud(tmp.name)
        highest_result = 0
        for file in os.listdir(os.path.join(settings.MEDIA_ROOT, authorization)):
            if file == "source.ply":
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
        return JsonResponse({'success': True, 'fitness': highest_result}, status=200)
    except Exception as e:
        log_error("authenticatePointCloud", extra={"message": f"Authentication failed: {e}"})
        return JsonResponse({'error': 'No registered point cloud.'}, status=400)

@csrf_exempt
def authenticate_two(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request'}, status=400)
    authorization = request.headers.get('Authorization')
    authMethod = request.headers.get('AuthorizationMethod')
    if not authorization:
        return JsonResponse({'error': 'Unauthorized'}, status=401)
    if 'files' not in request.FILES or len(request.FILES.getlist('files')) != 2:
        return JsonResponse({'error': 'Invalid number of files were uploaded.'}, status=400)
    if not authMethod:
        authMethod = "ICP"
    start = time()
    sourceFile = request.FILES.getlist('sourceFile')
    targetFile = request.FILES.getlist('targetFile')
    threshold = 0.7
    try:
        # Use tempfile to read both uploaded files into Open3D point clouds
        source_bytes = sourceFile[0].read()
        target_bytes = targetFile[0].read()
        with tempfile.NamedTemporaryFile(suffix=".ply") as tmp_source, tempfile.NamedTemporaryFile(suffix=".ply") as tmp_target:
            tmp_source.write(source_bytes)
            tmp_source.flush()
            tmp_target.write(target_bytes)
            tmp_target.flush()
            pcd_source = o3d.io.read_point_cloud(tmp_source.name)
            pcd_target = o3d.io.read_point_cloud(tmp_target.name)
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
        return JsonResponse({'success': True, 'fitness': result}, status=200)
    except Exception as e:
        log_error("authenticatePointCloud", extra={"message": f"Authentication failed: {e}"})
        return JsonResponse({'error': 'No registered point cloud.'}, status=400)