#!/bin/bash
# Script to copy trained model files from cluster to webapp

# After training on cluster, run this script to copy files to webapp
# Update these paths to match your cluster output directory

CLUSTER_OUTPUT="/path/to/your/cluster/outputs"
WEBAPP_MODELS="./webapp/models"

echo "Copying trained model files to webapp..."

# Copy model weights
cp "${CLUSTER_OUTPUT}/bird_attributes_model.pth" "${WEBAPP_MODELS}/"
echo "✓ Model weights copied"

# Copy calibrated thresholds
cp "${CLUSTER_OUTPUT}/optimal_thresholds.json" "${WEBAPP_MODELS}/"
echo "✓ Thresholds copied"

# Copy species centroids
cp "${CLUSTER_OUTPUT}/species_centroids.npy" "${WEBAPP_MODELS}/"
echo "✓ Centroids copied"

echo "All files copied! Restart your webapp to use the trained model."
echo "The webapp will automatically detect and load the trained weights."
