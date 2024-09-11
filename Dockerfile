# Use a Python image with OpenGL libraries pre-installed
FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu22.04

# Install Python 3.10 and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Set the working directory in the container
WORKDIR /worldauthdjango

# Copy the current directory contents into the container at /worldauthdjango
COPY . /worldauthdjango

# Install any necessary dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 to the outside world
EXPOSE 8000

# Command to run Django's development server
CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:8000", "worldauthdjango.wsgi:application"]

