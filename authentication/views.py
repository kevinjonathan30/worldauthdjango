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
import zipfile
from io import BytesIO
import shutil

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
        user_folder = os.path.join(settings.MEDIA_ROOT, authorization)
        os.makedirs(user_folder, exist_ok=True)
        pcd_list = os.listdir(user_folder)
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
        file_path = os.path.join(settings.MEDIA_ROOT, authorization, file_request)
        if os.path.exists(file_path):
            storage.delete(file_path)
        
        dir_path = os.path.join(settings.MEDIA_ROOT, authorization)
        remaining_files = [
            f for f in os.listdir(dir_path) if not f.startswith('.')
        ]
        if not remaining_files:
            shutil.rmtree(dir_path, ignore_errors=True)
        
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
        return JsonResponse({'success': True, 'fitness': highest_result}, status=200)
    except Exception as e:
        log_error("authenticatePointCloud", extra={"message": f"Authentication failed: {e}"})
        return JsonResponse({'error': 'No registered point cloud.'}, status=400)
    finally:
        storage.delete(sourcePath)

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
        return JsonResponse({'success': True, 'fitness': result}, status=200)
    except Exception as e:
        log_error("authenticatePointCloud", extra={"message": f"Authentication failed: {e}"})
        return JsonResponse({'error': 'No registered point cloud.'}, status=400)
    finally:
        storage.delete(sourcePath)
        storage.delete(targetPath) 

@csrf_exempt
def save_data(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request'}, status=400)
    authorization = request.headers.get('Authorization')
    if not authorization:
        return JsonResponse({'error': 'Unauthorized'}, status=401)
    if 'files' not in request.FILES or len(request.FILES.getlist('files')) != 1:
        return JsonResponse({'error': 'Invalid number of files were uploaded.'}, status=400)
    zip_file = request.FILES.getlist('files')[0]
    original_file_name = zip_file.name
    
    # Generate the file path using the original file name
    user_folder = os.path.join(settings.MEDIA_ROOT, authorization)
    os.makedirs(user_folder, exist_ok=True)
    file_path = os.path.join(user_folder, original_file_name)
    
    try:
        # Save the uploaded file
        with open(file_path, 'wb') as f:
            f.write(zip_file.read())
        return JsonResponse({'success': True}, status=200)
    except Exception as e:
        log_error("saveData", extra={"message": f"Failed to save data: {e}"})
        return JsonResponse({'success': False, 'error': 'Failed to save data.'}, status=500)
    
@csrf_exempt
def download_all(request):
    # Path to the root media directory
    media_root = settings.MEDIA_ROOT
    
    # Check if the media root exists
    if not os.path.exists(media_root):
        return JsonResponse({'error': 'Media folder does not exist.'}, status=404)
    
    # Create a ZIP file in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(media_root):
            for file in files:
                file_path = os.path.join(root, file)
                # Add file to the ZIP archive with a relative path
                zip_file.write(file_path, os.path.relpath(file_path, media_root))
    
    zip_buffer.seek(0)  # Rewind the buffer to the beginning
    
    # Prepare the response
    response = HttpResponse(zip_buffer, content_type='application/zip')
    response['Content-Disposition'] = 'attachment; filename="all_media_data.zip"'
    return response