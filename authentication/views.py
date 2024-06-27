from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.conf import settings
from .storage import OverwriteStorage
import open3d as o3d  # Open3D ライブラリ (Library)
from authentication import icp # Iterative closest points
from authentication import ply_helper
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
def delete_point_cloud(request, filename=None):
    if request.method == 'DELETE':
        authorization = request.headers.get('Authorization')
        if filename == None:
            return JsonResponse({'error': 'Invalid filename'}, status=400)
    
        storage = OverwriteStorage()
        try:
            storage.delete(authorization + "/" + filename)
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

        try:
            storage.save(authorization + "/" + files[0].name, ContentFile(files[0].read()))
            source = o3d.io.read_point_cloud(settings.MEDIA_ACCESS_URL + authorization + "/" + files[0].name)
            ply_helper.convert_ply_pcd(source, "source.pcd", authorization)
            # Clear data after converting
            storage.delete(authorization + "/" + files[0].name)

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

        storage.save(authorization + "/" + files[0].name, ContentFile(files[0].read()))

        try:
            # Convert to PCD first
            target = o3d.io.read_point_cloud(settings.MEDIA_ACCESS_URL + authorization + "/" + files[0].name)
            ply_helper.convert_ply_pcd(target, "target.pcd", authorization)
            # 入力データ (Input data)
            source = o3d.io.read_point_cloud(settings.MEDIA_ACCESS_URL + authorization + "/" + "source.pcd")
            target = o3d.io.read_point_cloud(settings.MEDIA_ACCESS_URL + authorization + "/" + "target.pcd")
            result = icp.execute_icp(source, target)
        except:
            return JsonResponse({'error': 'No registered point cloud.'}, status=400)
        finally:
            # Clear data after authenticating
            storage.delete(authorization + "/" + files[0].name)
            storage.delete(authorization + "/" + "target.pcd")

        if result >= 0.95:
            return JsonResponse({'success': True, 'fitness': result }, status=200)
        return JsonResponse({'success': False, 'fitness': result }, status=200)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)