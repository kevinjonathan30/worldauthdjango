from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.conf import settings

from .models import AuthenticationMethod
from .storage import OverwriteStorage

from authentication.icp import icp
from authentication.pointnetlk import pointnetlk
from authentication.fmr import fmr
from authentication.deepgmr import deepgmr

from authentication.helper import write_point_cloud, get_method_config

import os
from datetime import datetime
from time import time
from logger import log_debug, log_info, log_error

def index(request):
    log_debug(
        "accessIndex",
        extra={
            "message": "GET OUT OF HERE."
        }
    )
    return HttpResponse("GET OUT OF HERE.")

# curl -X GET http://127.0.0.1:8000/api/get_point_cloud/
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
        log_error(
            "getPointCloud",
            extra={
                "message": f"Failed to list point clouds: {e}"
            }
        )
        return JsonResponse({'success': True, 'data': []}, status=200)

# curl -X DELETE http://127.0.0.1:8000/api/delete_point_cloud/
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
        log_error(
            "deletePointCloud",
            extra={
                "message": f"Failed to delete point cloud: {e}"
            }
        )
        return JsonResponse({'success': False, 'error': 'Failed to delete Point Cloud.'}, status=500)

# curl -X POST -F 'files=@/path/to/file.ply' http://127.0.0.1:8000/api/register/
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
        log_error(
            "registerPointCloud",
            extra={
                "message": f"Failed to register point cloud: {e}"
            }
        )
        return JsonResponse({'success': False, 'error': 'Failed to register Point Cloud.'}, status=500)

# curl -X POST -F 'files=@/path/to/file.ply' http://127.0.0.1:8000/api/authenticate/
@csrf_exempt
def authenticate(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request'}, status=400)
    
    authorization = request.headers.get('Authorization')
        
    if not authorization:
        return JsonResponse({'error': 'Unauthorized'}, status=401)
    if 'files' not in request.FILES or len(request.FILES.getlist('files')) != 1:
        return JsonResponse({'error': 'Invalid number of files were uploaded.'}, status=400)
    
    method = get_method_config()
    start = time()
        
    files = request.FILES.getlist('files')
    storage = OverwriteStorage()
    source = os.path.join(settings.MEDIA_ROOT, authorization, "source.ply")

    threshold=0.7
        
    try:
        write_point_cloud(source, ContentFile(files[0].read()))
            
        highest_result = 0
        for file in os.listdir(os.path.join(settings.MEDIA_ROOT, authorization)):
            if file == "source.ply":
                continue
            target = os.path.join(settings.MEDIA_ROOT, authorization, file)

            if method == AuthenticationMethod.ICP:
                threshold=0.545
                result = icp.execute_icp(source, target, threshold=threshold)
            elif method == AuthenticationMethod.FGR:
                threshold=0.54
                result = icp.execute_icp(source, target, fast_registration=True, threshold=threshold)
            elif method == AuthenticationMethod.PNLK:
                threshold=0.58
                result = pointnetlk.execute_pointnetlk(source, target, threshold=threshold)
            elif method == AuthenticationMethod.FMR:
                threshold=0.66
                result = fmr.execute_fmr(source, target, threshold=threshold)
            elif method == AuthenticationMethod.DGMR:
                threshold=0.675
                result = deepgmr.execute_deepgmr(source, target, threshold=threshold)
            
            highest_result = max(highest_result, result)
            if highest_result == 1:
                break
            
        authentication_time = time() - start
        log_info(
            "authenticatePointCloud",
            extra={
                "method": method.value,
                "fitness": highest_result,
                "threshold": threshold,
                "authTime": authentication_time
            }
        )
        return JsonResponse({'success': True, 'fitness': highest_result}, status=200)
    except Exception as e:
        log_error(
            "authenticatePointCloud",
            extra={
                "message": f"Authentication failed: {e}"
            }
        )
        return JsonResponse({'error': 'No registered point cloud.'}, status=400)
    finally:
        storage.delete(source)
