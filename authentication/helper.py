import os

import open3d as o3d
import numpy as np
from io import BytesIO, StringIO

from django.conf import settings
import shutil

def write_point_cloud(filename, fileContent, format='ply', write_ascii=False, compressed=True):
    """
    Write a point cloud to disk from various input types.

    fileContent may be:
    - a filesystem path (str)
    - bytes or bytearray
    - a Django UploadedFile / ContentFile (file-like with .read())
    - a file-like object (binary or text)

    This function attempts to read numeric point data (x y z per line). If the
    content is empty or not plain-text numeric, a clear exception is raised.
    """
    # Normalize input to bytes
    data_bytes = None
    if isinstance(fileContent, str) and os.path.exists(fileContent):
        # treat as path
        with open(fileContent, 'rb') as f:
            data_bytes = f.read()
    elif isinstance(fileContent, (bytes, bytearray)):
        data_bytes = bytes(fileContent)
    else:
        # file-like: try to call read()
        try:
            read_result = fileContent.read()
            # UploadedFile.read() returns bytes; ContentFile may return bytes too
            if isinstance(read_result, str):
                data_bytes = read_result.encode('utf-8')
            else:
                data_bytes = read_result
        except Exception:
            # Last resort: try converting to str then to bytes
            try:
                data_bytes = str(fileContent).encode('utf-8')
            except Exception:
                raise ValueError("Unsupported fileContent type for write_point_cloud")

    if not data_bytes:
        raise ValueError("Empty file content - nothing to write as point cloud")

    # Try to parse as ASCII numeric list (x y z per line)
    points_np = None
    text_stream = None
    try:
        # decode, but be tolerant of null bytes
        decoded = data_bytes.decode('utf-8', errors='ignore')
        text_stream = StringIO(decoded)
        # np.loadtxt works with file-like objects
        points_np = np.loadtxt(text_stream)
    except Exception:
        # fallback: try fromstring on the raw bytes (whitespace separated)
        try:
            points_np = np.fromstring(data_bytes.decode('utf-8', errors='ignore'), sep=' ')
            if points_np.size == 0:
                points_np = None
            else:
                # reshape if possible to (-1,3)
                if points_np.size % 3 == 0:
                    points_np = points_np.reshape((-1, 3))
                else:
                    # can't reshape meaningfully
                    points_np = None
        except Exception:
            points_np = None

    if points_np is None or points_np.size == 0:
        # It might be a binary PLY or other binary format; try using Open3D reader
        # by writing bytes to a temporary BytesIO and asking Open3D to read it.
        # Open3D doesn't read from bytes directly, so write to a temp file.
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(data_bytes)
            tmp.flush()
            tmp_path = tmp.name
        try:
            pcd = o3d.io.read_point_cloud(tmp_path)
            if pcd.is_empty():
                raise ValueError("Could not parse point cloud from uploaded content")
            # ensure directory exists
            directory = os.path.dirname(filename)
            if not os.path.exists(directory):
                os.makedirs(directory)
            # write using Open3D (it will re-format as needed)
            o3d.io.write_point_cloud(filename=filename, pointcloud=pcd, format=format, write_ascii=write_ascii, compressed=compressed)
            return
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    # At this point we have numeric points in points_np
    # Ensure shape is (N,3)
    if points_np.ndim == 1 and points_np.size % 3 == 0:
        points_np = points_np.reshape((-1, 3))
    if points_np.ndim != 2 or points_np.shape[1] != 3:
        raise ValueError("Point data must be Nx3 floats (x y z per row)")

    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(points_np)

    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    o3d.io.write_point_cloud(filename=filename, pointcloud=ply, format=format, write_ascii=write_ascii, compressed=compressed)

def delete_parent_directory(authorization):
    dir_path = os.path.join(settings.MEDIA_ROOT, authorization)
    remaining_files = [
        f for f in os.listdir(dir_path) if not f.startswith('.')
    ]
    if not remaining_files:
        shutil.rmtree(dir_path, ignore_errors=True)