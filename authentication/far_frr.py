import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Set backend to 'Agg' for non-GUI usage
matplotlib.use('Agg')

from django.conf import settings
from .models import AuthenticationMethod
from authentication.icp import icp
from authentication.pointnetlk import pointnetlk 
from authentication.fmr import fmr 
from authentication.deepgmr import deepgmr 
from datetime import datetime
import os

# Simulated function to compare point clouds and return a fitness score
def authenticate_point_cloud(source_ply, target_ply, threshold):
    method = AuthenticationMethod.FMR

    source = os.path.join(settings.MEDIA_ROOT, source_ply)
    target = os.path.join(settings.MEDIA_ROOT, target_ply)
        
    if method == AuthenticationMethod.ICP:
        # print("Using ICP Point-to-Plane & RANSAC")
        result = icp.execute_icp(source, target, threshold=threshold)
    elif method == AuthenticationMethod.FGR:
        # print("Using Fast Global Registration")
        result = icp.execute_icp(source, target, fast_registration=True, threshold=threshold)
    elif method == AuthenticationMethod.PNLK:
        # print("Using PointNetLK")
        result = pointnetlk.execute_pointnetlk(source, target, threshold=threshold)
    elif method == AuthenticationMethod.FMR:
        # print("Using Feature-metric Registration")
        result = fmr.execute_fmr(source, target, threshold=threshold)
    elif method == AuthenticationMethod.DGMR:
        # print("Using DeepGMR")
        result = deepgmr.execute_deepgmr(source, target, threshold=threshold)

    return result

# Directories containing the genuine and imposter point clouds
genuine_dir = 'genuine'
imposter_dir = 'imposter'

# List to store FAR and FRR values at different thresholds
thresholds = np.arange(0, 1.05, 0.05)  # Example thresholds from 0 to 1
far_values = []
frr_values = []

# Function to calculate FAR and FRR at a given threshold
def calculate_far_frr(threshold):
    false_acceptances = 0
    false_rejections = 0
    genuine_attempts = 0
    imposter_attempts = 0

    # Check genuine point clouds (expect success)
    for genuine_file in os.listdir(os.path.join(settings.MEDIA_ROOT, genuine_dir)):
        genuine_attempts += 1
        for target_file in os.listdir(os.path.join(settings.MEDIA_ROOT, genuine_dir)):
            if genuine_file != target_file:
                source_ply = os.path.join(genuine_dir, genuine_file)
                target_ply = os.path.join(genuine_dir, target_file)
                if authenticate_point_cloud(source_ply, target_ply, threshold) < 1:
                    false_rejections += 1  # Genuine was rejected

    # Check imposter point clouds (expect failure)
    for imposter_file in os.listdir(os.path.join(settings.MEDIA_ROOT, imposter_dir)):
        imposter_attempts += 1
        for target_file in os.listdir(os.path.join(settings.MEDIA_ROOT, genuine_dir)):
            source_ply = os.path.join(os.path.join(settings.MEDIA_ROOT, imposter_dir), imposter_file)
            target_ply = os.path.join(os.path.join(settings.MEDIA_ROOT, genuine_dir), target_file)
            if authenticate_point_cloud(source_ply, target_ply, threshold) >= 1:
                false_acceptances += 1  # Imposter was accepted

    # Calculate FAR and FRR
    far = false_acceptances / imposter_attempts if imposter_attempts > 0 else 0
    frr = false_rejections / genuine_attempts if genuine_attempts > 0 else 0
    return far, frr

# Function to find the point of intersection (smallest difference between FAR and FRR)
def find_intersection(far_values, frr_values, thresholds):
    differences = np.abs(np.array(far_values) - np.array(frr_values))
    min_index = np.argmin(differences)
    return thresholds[min_index], far_values[min_index], frr_values[min_index]

# Calculate FAR and FRR for each threshold
def execute_far_frr():
    for threshold in thresholds:
        far, frr = calculate_far_frr(threshold)
        far_values.append(far)
        frr_values.append(frr)

    # Find the intersection point (threshold where FAR and FRR are closest)
    intersection_threshold, far_at_intersection, frr_at_intersection = find_intersection(far_values, frr_values, thresholds)

    # Plotting the FAR and FRR curves
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, far_values, label="FAR", marker='o')
    plt.plot(thresholds, frr_values, label="FRR", marker='x')

    # Mark the intersection point
    plt.scatter(intersection_threshold, far_at_intersection, color='red', zorder=5)
    plt.text(intersection_threshold, far_at_intersection + 0.1, f'Intersection\nThreshold: {intersection_threshold:.3f}', color='red', ha='center')

    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title('FAR and FRR vs. Threshold')
    plt.legend()
    plt.grid(True)

    # Save the plot as a file
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    plt.savefig(os.path.join(settings.MEDIA_ROOT, f'far_frr_intersection_plot-{dt_string}.png'))

    # Print the point of intersection to the console
    print(f"Intersection at Threshold: {intersection_threshold:.3f}, FAR: {far_at_intersection:.3f}, FRR: {frr_at_intersection:.3f}")

    return intersection_threshold