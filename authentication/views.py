from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.conf import settings
from .storage import OverwriteStorage
import open3d as o3d  # Open3D ライブラリ (Library)
from authentication import icp # Iterative closest points
import os
import logging

logger = logging.getLogger(__name__)

def index(request):
    return HttpResponse("GET OUT OF HERE.")

# curl -X GET http://127.0.0.1:8000/api/get_point_cloud/
@csrf_exempt
def get_point_cloud(request):
    if request.method == 'GET':
        authorization = request.headers.get('Authorization')
        if not authorization:
            return JsonResponse({'error': 'Unauthorized'}, status=401)
        
        try:
            pcd_list = os.listdir(os.path.join(settings.MEDIA_ACCESS_URL, authorization))
            return JsonResponse({'success': True, 'data': pcd_list}, status=200)
        except Exception as e:
            logger.error(f"Failed to list point clouds: {e}")
            return JsonResponse({'success': True, 'data': []}, status=200)
    return JsonResponse({'error': 'Invalid request'}, status=400)

# curl -X DELETE http://127.0.0.1:8000/api/delete_point_cloud/
@csrf_exempt
def delete_point_cloud(request):
    if request.method == 'DELETE':
        authorization = request.headers.get('Authorization')
        file_request = request.headers.get('FileRequest')
        
        if not authorization:
            return JsonResponse({'error': 'Unauthorized'}, status=401)
        if not file_request:
            return JsonResponse({'error': 'Invalid filename'}, status=400)
        
        storage = OverwriteStorage()
        try:
            storage.delete(os.path.join(authorization, file_request))
            remaining_files = os.listdir(os.path.join(settings.MEDIA_ACCESS_URL, authorization))
            if not remaining_files:
                storage.delete(authorization)
            return JsonResponse({'success': True}, status=200)
        except Exception as e:
            logger.error(f"Failed to delete point cloud: {e}")
            return JsonResponse({'success': False, 'error': 'Failed to delete Point Cloud.'}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=400)

# curl -X POST -F 'files=@/path/to/file.ply' http://127.0.0.1:8000/api/register/
@csrf_exempt
def register(request):
    if request.method == 'POST':
        authorization = request.headers.get('Authorization')
        
        if not authorization:
            return JsonResponse({'error': 'Unauthorized'}, status=401)
        if 'files' not in request.FILES or len(request.FILES.getlist('files')) != 1:
            return JsonResponse({'error': 'Invalid number of files were uploaded.'}, status=400)
        
        files = request.FILES.getlist('files')
        storage = OverwriteStorage()
        room_sequence = ["Room_1.ply", "Room_2.ply", "Room_3.ply"]
        
        try:
            pcd_list = os.listdir(os.path.join(settings.MEDIA_ACCESS_URL, authorization))
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
            storage.save(os.path.join(authorization, next_room), ContentFile(files[0].read()))
            return JsonResponse({'success': True}, status=200)
        except Exception as e:
            logger.error(f"Failed to register point cloud: {e}")
            return JsonResponse({'success': False, 'error': 'Failed to register Point Cloud.'}, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

# curl -X POST -F 'files=@/path/to/file.ply' http://127.0.0.1:8000/api/authenticate/
@csrf_exempt
def authenticate(request):
    if request.method == 'POST':
        authorization = request.headers.get('Authorization')
        
        if not authorization:
            return JsonResponse({'error': 'Unauthorized'}, status=401)
        if 'files' not in request.FILES or len(request.FILES.getlist('files')) != 1:
            return JsonResponse({'error': 'Invalid number of files were uploaded.'}, status=400)
        
        files = request.FILES.getlist('files')
        storage = OverwriteStorage()
        target_file_path = os.path.join(authorization, "target.ply")
        
        try:
            storage.save(target_file_path, ContentFile(files[0].read()))
            target = o3d.io.read_point_cloud(os.path.join(settings.MEDIA_ACCESS_URL, target_file_path))
            
            highest_result = 0
            for file in os.listdir(os.path.join(settings.MEDIA_ACCESS_URL, authorization)):
                if file == "target.ply":
                    continue
                source = o3d.io.read_point_cloud(os.path.join(settings.MEDIA_ACCESS_URL, authorization, file))
                result = icp.execute_icp(source, target)
                highest_result = max(highest_result, result)
            
            if highest_result == 1:
                return JsonResponse({'success': True, 'fitness': highest_result}, status=200)
            return JsonResponse({'success': False, 'fitness': highest_result}, status=200)
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return JsonResponse({'error': 'No registered point cloud.'}, status=400)
        finally:
            storage.delete(target_file_path)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)
