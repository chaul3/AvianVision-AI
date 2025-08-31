#!/bin/bash

# Quick fix for CUB dataset parsing issues on cluster
# This script addresses the "Expected 5 fields, saw 7" error

echo "🔧 CUB Dataset Quick Fix for Cluster"
echo "===================================="

# Find the dataset path from config
DATASET_PATH=""
if [ -f "config_demo.yaml" ]; then
    DATASET_PATH=$(grep "cub_root:" config_demo.yaml | cut -d'"' -f2)
elif [ -f "config_cluster.yaml" ]; then
    DATASET_PATH=$(grep "cub_root:" config_cluster.yaml | cut -d'"' -f2)
else
    echo "❌ No config file found. Please specify dataset path:"
    read -p "Enter CUB dataset path: " DATASET_PATH
fi

echo "📁 Dataset path: $DATASET_PATH"

ATTR_FILE="$DATASET_PATH/attributes/image_attribute_labels.txt"

if [ ! -f "$ATTR_FILE" ]; then
    echo "❌ Attribute file not found: $ATTR_FILE"
    exit 1
fi

echo "🔍 Checking attribute file..."

# Check for problematic lines
echo "📊 File statistics:"
wc -l "$ATTR_FILE"

echo "🔍 Looking for lines with incorrect field count..."
awk 'NF != 4 {print "Line " NR ": " NF " fields - " $0}' "$ATTR_FILE" | head -10

# Count problematic lines
PROBLEM_LINES=$(awk 'NF != 4 {count++} END {print count+0}' "$ATTR_FILE")
echo "⚠️  Found $PROBLEM_LINES lines with incorrect field count"

if [ "$PROBLEM_LINES" -gt 0 ]; then
    echo ""
    echo "🔧 Fixing the file..."
    
    # Create backup
    cp "$ATTR_FILE" "$ATTR_FILE.backup"
    echo "📋 Backup created: $ATTR_FILE.backup"
    
    # Fix the file by keeping only first 4 fields
    awk '{print $1, $2, $3, $4}' "$ATTR_FILE.backup" > "$ATTR_FILE.fixed"
    
    # Validate the fix
    FIXED_PROBLEMS=$(awk 'NF != 4 {count++} END {print count+0}' "$ATTR_FILE.fixed")
    
    if [ "$FIXED_PROBLEMS" -eq 0 ]; then
        mv "$ATTR_FILE.fixed" "$ATTR_FILE"
        echo "✅ File fixed successfully!"
        echo "📊 All lines now have exactly 4 fields"
    else
        echo "❌ Fix failed, $FIXED_PROBLEMS lines still problematic"
        rm "$ATTR_FILE.fixed"
    fi
else
    echo "✅ File looks good, no fixes needed"
fi

echo ""
echo "🚀 You can now run the pipeline:"
echo "python evaluate_pipeline.py --config config_demo.yaml"
