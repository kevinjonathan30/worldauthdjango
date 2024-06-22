from django.core.files.storage import FileSystemStorage

class OverwriteStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        # If the file exists, delete it before returning the name
        if self.exists(name):
            self.delete(name)
        return name
