from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.conf import settings
from .storage import OverwriteStorage
import open3d as o3d  # Open3D ライブラリ (Library)
from authentication import icp # Iterative closest points
from authentication import ply_helper


def index(request):
    return HttpResponse("GET OUT OF HERE.")

# curl -X POST -F 'files=@/Users/kevinjonathan/Downloads/PointCloud_0006-06-19-15-20-00.ply' http://127.0.0.1:8000/api/register/
def register(request):
    if request.method == 'POST':
        if 'files' not in request.FILES or len(request.FILES.getlist('files')) != 1:
            return JsonResponse({'error': 'Invalid number of files were uploaded.'}, status=400)
        
        files = request.FILES.getlist('files')
        storage = OverwriteStorage()

        try:
            storage.save(files[0].name, ContentFile(files[0].read()))
            source = o3d.io.read_point_cloud(settings.MEDIA_ACCESS_URL + files[0].name)
            ply_helper.convert_ply_pcd(source, "source.pcd")
            
            # Clear data after converting
            storage.delete(files[0].name)

            return JsonResponse({'success': True }, status=200)
        except:
            return JsonResponse({'success': False, 'error': 'Failed to register Point Cloud.' }, status=200)
    
    return JsonResponse({'error': 'Invalid request.'}, status=400)

# curl -X POST -F 'files=@/Users/kevinjonathan/Downloads/PointCloud_0006-06-19-15-20-16.ply' http://127.0.0.1:8000/api/authenticate/
@csrf_exempt
def authenticate(request):
    if request.method == 'POST':
        if 'files' not in request.FILES or len(request.FILES.getlist('files')) != 1:
            return JsonResponse({'error': 'Invalid number of files were uploaded.'}, status=400)
        
        files = request.FILES.getlist('files')
        storage = OverwriteStorage()

        storage.save(files[0].name, ContentFile(files[0].read()))

        # Convert to PCD first
        target = o3d.io.read_point_cloud(settings.MEDIA_ACCESS_URL + files[0].name)
        ply_helper.convert_ply_pcd(target, "target.pcd")
        # 入力データ (Input data)
        source = o3d.io.read_point_cloud(settings.MEDIA_ACCESS_URL + "source.pcd")
        target = o3d.io.read_point_cloud(settings.MEDIA_ACCESS_URL + "target.pcd")
        result = icp.execute_icp(source, target)

        # Clear data after authenticating
        storage.delete(files[0].name)
        storage.delete("target.pcd")

        if result >= 0.95:
            return JsonResponse({'success': True, 'fitness': result }, status=200)
        return JsonResponse({'success': False, 'fitness': result }, status=200)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)