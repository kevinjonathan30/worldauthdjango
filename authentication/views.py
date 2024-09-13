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
from authentication.helper import write_point_cloud
import os
import logging
from datetime import datetime
from time import time


logger = logging.getLogger('authentication')

def index(request):
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
        logger.error(f"Failed to list point clouds: {e}")
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
        logger.error(f"Failed to delete point cloud: {e}")
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
    room_sequence = ["Room_1.ply", "Room_2.ply", "Room_3.ply"]
        
    try:
        pcd_list = os.listdir(os.path.join(settings.MEDIA_ROOT, authorization))
        rooms_present = {room: room in pcd_list for room in room_sequence}
    except Exception as e:
        logger.error(f"Failed to list rooms: {e}")
        rooms_present = {room: False for room in room_sequence}
        
    next_room = None
    for room in room_sequence:
        if not rooms_present[room]:
            next_room = room
            break
        
    if not next_room:
        return JsonResponse({'success': False, 'error': 'Exceeded registration limit.'}, status=400)
        
    try:
        file_path = os.path.join(settings.MEDIA_ROOT, authorization, next_room)

        write_point_cloud(file_path, ContentFile(files[0].read()))
        return JsonResponse({'success': True}, status=200)
    except Exception as e:
        logger.error(f"Failed to register point cloud: {e}")
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
    
    method = AuthenticationMethod.FMR
    start = time()
        
    files = request.FILES.getlist('files')
    storage = OverwriteStorage()
    source = os.path.join(settings.MEDIA_ROOT, authorization, "source.ply")
        
    try:
        write_point_cloud(source, ContentFile(files[0].read()))
            
        highest_result = 0
        for file in os.listdir(os.path.join(settings.MEDIA_ROOT, authorization)):
            if file == "source.ply":
                continue
            target = os.path.join(settings.MEDIA_ROOT, authorization, file)

            if method == AuthenticationMethod.ICP:
                logger.info("Using ICP Point-to-Plane & RANSAC")
                result = icp.execute_icp(source, target)
            elif method == AuthenticationMethod.FGR:
                logger.info("Using Fast Global Registration")
                result = icp.execute_icp(source, target, fast_registration=True)
            elif method == AuthenticationMethod.PNLK:
                logger.info("Using PointNetLK")
                result = pointnetlk.execute_pointnetlk(source, target)
            elif method == AuthenticationMethod.FMR:
                logger.info("Using Feature-metric Registration")
                result = fmr.execute_fmr(source, target)
            elif method == AuthenticationMethod.DGMR:
                logger.info("Using DeepGMR")
                result = deepgmr.execute_deepgmr(source, target)
            
            if result == 1:
                authentication_time = time() - start
                logger.info(f'Authentication time: {authentication_time}')
                return JsonResponse({'success': True, 'fitness': result}, status=200)
            highest_result = max(highest_result, result)
            
        authentication_time = time() - start
        logger.info(f'Authentication time: {authentication_time}')
        return JsonResponse({'success': False, 'fitness': highest_result}, status=200)
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return JsonResponse({'error': 'No registered point cloud.'}, status=400)
    finally:
        storage.delete(source)


# curl -X POST -F 'files=@/path/to/file.ply' http://127.0.0.1:8000/api/collect/
@csrf_exempt
def collect(request):
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
        logger.error(f"Failed to register point cloud: {e}")
        return JsonResponse({'success': False, 'error': 'Failed to register Point Cloud.'}, status=500)
