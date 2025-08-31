#!/usr/bin/env python3
"""
Quick CUB Dataset File Inspector

This script inspects the CUB dataset files to understand their exact format
and help diagnose parsing issues.
"""

import os
import sys


def inspect_file(filepath, max_lines=10, name="file"):
    """Inspect a text file and show its structure."""
    print(f"\n📁 Inspecting {name}: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return
    
    try:
        file_size = os.path.getsize(filepath)
        print(f"📊 File size: {file_size:,} bytes")
        
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = []
            field_counts = {}
            total_lines = 0
            
            for line_num, line in enumerate(f, 1):
                total_lines = line_num
                line = line.strip()
                
                if not line:
                    continue
                
                if line_num <= max_lines:
                    lines.append((line_num, line))
                
                # Count fields
                fields = len(line.split())
                field_counts[fields] = field_counts.get(fields, 0) + 1
                
                # Stop after reading enough for analysis
                if line_num > 100000:  # Don't read more than 100k lines for inspection
                    break
        
        print(f"📊 Total lines: {total_lines:,}")
        print(f"📊 Field count distribution:")
        for fields, count in sorted(field_counts.items()):
            print(f"   {fields} fields: {count:,} lines")
        
        print(f"📝 First {len(lines)} lines:")
        for line_num, line in lines:
            fields = line.split()
            print(f"   Line {line_num:3d} ({len(fields)} fields): {line[:80]}")
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")


def main():
    print("🔍 CUB Dataset File Inspector")
    print("=" * 40)
    
    # Try to find dataset path
    dataset_paths = [
        "data/CUB_200_2011",
        "/home/leph/AvianVision-AI/data/CUB_200_2011",
        "CUB_200_2011"
    ]
    
    dataset_path = None
    for path in dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if not dataset_path:
        print("Enter CUB dataset path:")
        dataset_path = input().strip()
    
    print(f"📁 Using dataset path: {dataset_path}")
    
    # Files to inspect
    files_to_check = [
        ("images.txt", os.path.join(dataset_path, "images.txt")),
        ("classes.txt", os.path.join(dataset_path, "classes.txt")),
        ("image_class_labels.txt", os.path.join(dataset_path, "image_class_labels.txt")),
        ("train_test_split.txt", os.path.join(dataset_path, "train_test_split.txt")),
        ("image_attribute_labels.txt", os.path.join(dataset_path, "attributes", "image_attribute_labels.txt")),
    ]
    
    for name, filepath in files_to_check:
        inspect_file(filepath, max_lines=5, name=name)
    
    print(f"\n🎯 Quick Analysis:")
    
    # Check specific attribute file issues
    attr_file = os.path.join(dataset_path, "attributes", "image_attribute_labels.txt")
    if os.path.exists(attr_file):
        print(f"\n🔍 Detailed analysis of image_attribute_labels.txt:")
        
        # Check for specific line that caused the error
        try:
            with open(attr_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if line_num == 709498:  # The problematic line from the error
                        print(f"Line 709498: {line.strip()}")
                        print(f"Fields: {len(line.split())}")
                        break
                    if line_num > 709500:  # Don't read too far
                        break
        except Exception as e:
            print(f"Could not read line 709498: {e}")
        
        # Count problematic lines
        try:
            with open(attr_file, 'r') as f:
                problem_count = 0
                for line_num, line in enumerate(f, 1):
                    fields = len(line.strip().split())
                    if fields != 4:
                        problem_count += 1
                        if problem_count <= 5:  # Show first 5 problems
                            print(f"Problem line {line_num}: {fields} fields - {line.strip()[:60]}")
                
                print(f"\n📊 Total problematic lines: {problem_count}")
        except Exception as e:
            print(f"Error analyzing problematic lines: {e}")


if __name__ == "__main__":
    main()
