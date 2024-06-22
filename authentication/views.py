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
    return HttpResponse("Hello, world. You're at the authentication index.")


@csrf_exempt
def authenticate(request):
    if request.method == 'POST':
        if 'files' not in request.FILES:
            return JsonResponse({'error': 'Invalid file numbers were uploaded.'}, status=400)
        
        files = request.FILES.getlist('files')
        file_urls = []
        storage = OverwriteStorage()

        for file in files:
            file_name = storage.save(file.name, ContentFile(file.read()))
            file_url = settings.MEDIA_URL + file_name
            file_urls.append(file_url)

        # Convert to PCD first
        current_directory = os.getcwd()
        print("Current Directory:", current_directory)
        source = o3d.io.read_point_cloud(settings.MEDIA_ACCESS_URL + files[0].name)
        target = o3d.io.read_point_cloud(settings.MEDIA_ACCESS_URL + files[1].name)
        ply_helper.convert_ply_pcd(source, "plyFile.pcd")
        ply_helper.convert_ply_pcd(target, "plyFile1.pcd")  
        # 入力データ (Input data)
        source = o3d.io.read_point_cloud(settings.MEDIA_ACCESS_URL + "plyFile.pcd")
        target = o3d.io.read_point_cloud(settings.MEDIA_ACCESS_URL + "plyFile1.pcd")
        result = icp.execute_icp(source, target)
        if result > 0.95:
            return JsonResponse({'result': True }, status=200)
        return JsonResponse({'result': False }, status=200)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)