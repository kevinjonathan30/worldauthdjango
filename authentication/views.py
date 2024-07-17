from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.conf import settings
from .storage import OverwriteStorage
import open3d as o3d  # Open3D ライブラリ (Library)
from authentication import icp # Iterative closest points
import os


def index(request):
    return HttpResponse("GET OUT OF HERE.")

# curl -X GET http://127.0.0.1:8000/api/get_point_cloud/
@csrf_exempt
def get_point_cloud(request):
    if request.method == 'GET':
        authorization = request.headers.get('Authorization')
        try:
            pcd_list = os.listdir(settings.MEDIA_ACCESS_URL + authorization)
            return JsonResponse({'success': True, 'data': pcd_list}, status=200)
        except:
            return JsonResponse({'success': True, 'data': []}, status=200)
    return JsonResponse({'error': 'Invalid request'}, status=400)

# curl -X DELETE http://127.0.0.1:8000/api/delete_point_cloud/
@csrf_exempt
def delete_point_cloud(request):
    if request.method == 'DELETE':
        authorization = request.headers.get('Authorization')
        fileRequest = request.headers.get('FileRequest')
        if fileRequest == None:
            return JsonResponse({'error': 'Invalid filename'}, status=400)
    
        storage = OverwriteStorage()
        try:
            storage.delete(authorization + "/" + fileRequest)
            if len(os.listdir(settings.MEDIA_ACCESS_URL + authorization)) == 0:
                storage.delete(authorization)
            return JsonResponse({'success': True}, status=200)
        except:
            return JsonResponse({'success': False, 'error': 'Failed to delete Point Cloud.' }, status=200)
    return JsonResponse({'error': 'Invalid request'}, status=400)

# curl -X POST -F 'files=@/Users/kevinjonathan/Downloads/PointCloud_0006-06-19-15-20-00.ply' http://127.0.0.1:8000/api/register/
@csrf_exempt
def register(request):
    if request.method == 'POST':
        authorization = request.headers.get('Authorization')
        if 'files' not in request.FILES or len(request.FILES.getlist('files')) != 1:
            return JsonResponse({'error': 'Invalid number of files were uploaded.'}, status=400)
        
        files = request.FILES.getlist('files')
        storage = OverwriteStorage()

        room_sequence = ["Room_1.ply", "Room_2.ply", "Room_3.ply"]
        try:
            pcd_list = os.listdir(settings.MEDIA_ACCESS_URL + authorization)
            rooms_present = {room: room in pcd_list for room in room_sequence}
        except:
            rooms_present = {room: False for room in room_sequence}
            
            # Determine the next room to add
        if not rooms_present["Room_1.ply"]:
            next_room = "Room_1.ply"
        elif not rooms_present["Room_2.ply"]:
            next_room = "Room_2.ply"
        elif not rooms_present["Room_3.ply"]:
            next_room = "Room_3.ply"
        else:
            next_room = None
            
        # Add the next room if necessary
        if next_room is None:
            return JsonResponse({'success': False, 'error': 'Exceeded registration limit.' }, status=200)
        
        try:
            storage.save(authorization + "/" + next_room, ContentFile(files[0].read()))

            return JsonResponse({'success': True }, status=200)
        except:
            return JsonResponse({'success': False, 'error': 'Failed to register Point Cloud.' }, status=200)
    
    return JsonResponse({'error': 'Invalid request.'}, status=400)

# curl -X POST -F 'files=@/Users/kevinjonathan/Downloads/PointCloud_0006-06-19-15-20-16.ply' http://127.0.0.1:8000/api/authenticate/
@csrf_exempt
def authenticate(request):
    if request.method == 'POST':
        authorization = request.headers.get('Authorization')
        if 'files' not in request.FILES or len(request.FILES.getlist('files')) != 1:
            return JsonResponse({'error': 'Invalid number of files were uploaded.'}, status=400)
        
        files = request.FILES.getlist('files')
        storage = OverwriteStorage()

        storage.save(authorization + "/target.ply", ContentFile(files[0].read()))
        target = o3d.io.read_point_cloud(settings.MEDIA_ACCESS_URL + authorization + "/target.ply")

        try:
            highestResult = 0
            for file in os.listdir(settings.MEDIA_ACCESS_URL + authorization):
                if file == "target.ply":
                    continue
                # 入力データ (Input data)
                source = o3d.io.read_point_cloud(settings.MEDIA_ACCESS_URL + authorization + "/" + file)
                result = icp.execute_icp(source, target)
                if result > highestResult:
                    highestResult = result

            if highestResult == 1:
                return JsonResponse({'success': True, 'fitness': highestResult }, status=200)
            return JsonResponse({'success': False, 'fitness': highestResult }, status=200)
        except:
            return JsonResponse({'error': 'No registered point cloud.'}, status=400)
        finally:
            storage.delete(authorization + "/target.ply")
    
    return JsonResponse({'error': 'Invalid request'}, status=400)