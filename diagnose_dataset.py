"""
Diagnostic script for CUB dataset parsing issues.

This script helps diagnose and fix common issues with CUB-200-2011 dataset files.
"""

import os
import pandas as pd
from pathlib import Path


def diagnose_file(file_path: str, expected_fields: int = 4, max_lines_to_check: int = 10):
    """Diagnose issues with a space-separated file."""
    print(f"\n🔍 Diagnosing file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    file_size = os.path.getsize(file_path)
    print(f"📊 File size: {file_size:,} bytes")
    
    issues = []
    line_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_count = line_num
                line = line.strip()
                
                if not line:
                    continue
                
                fields = line.split()
                
                if len(fields) != expected_fields:
                    issues.append({
                        'line': line_num,
                        'expected': expected_fields,
                        'found': len(fields),
                        'content': line[:100] + ('...' if len(line) > 100 else '')
                    })
                    
                    if len(issues) <= max_lines_to_check:
                        print(f"⚠️  Line {line_num}: Expected {expected_fields} fields, found {len(fields)}")
                        print(f"    Content: {line[:100]}")
                
                # Show first few lines as examples
                if line_num <= 3:
                    print(f"📝 Line {line_num}: {line[:100]}")
    
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    print(f"📊 Total lines: {line_count:,}")
    print(f"⚠️  Issues found: {len(issues)}")
    
    if issues:
        print(f"\n🔧 Most common issue types:")
        field_counts = {}
        for issue in issues:
            found = issue['found']
            field_counts[found] = field_counts.get(found, 0) + 1
        
        for fields, count in sorted(field_counts.items()):
            print(f"   {count} lines with {fields} fields")
    
    return len(issues) == 0


def fix_file(input_path: str, output_path: str, expected_fields: int = 4):
    """Fix a malformed file by keeping only the first N fields."""
    print(f"\n🔧 Fixing file: {input_path} -> {output_path}")
    
    fixed_lines = 0
    total_lines = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            total_lines = line_num
            line = line.strip()
            
            if not line:
                outfile.write('\n')
                continue
            
            fields = line.split()
            
            if len(fields) > expected_fields:
                # Keep only the first expected_fields
                fixed_line = ' '.join(fields[:expected_fields])
                outfile.write(fixed_line + '\n')
                fixed_lines += 1
            elif len(fields) < expected_fields:
                # Skip lines with too few fields
                print(f"⚠️  Skipping line {line_num} (too few fields): {line[:50]}")
                continue
            else:
                outfile.write(line + '\n')
    
    print(f"✅ Fixed {fixed_lines} lines out of {total_lines} total lines")
    return output_path


def main():
    """Main diagnostic function."""
    print("🔍 CUB Dataset File Diagnostic Tool")
    print("=" * 50)
    
    # Check for config file to find dataset path
    config_files = ['config_demo.yaml', 'config_cluster.yaml']
    dataset_path = None
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    dataset_path = config.get('data', {}).get('cub_root')
                    if dataset_path:
                        print(f"📁 Found dataset path in {config_file}: {dataset_path}")
                        break
            except Exception as e:
                print(f"⚠️  Could not read {config_file}: {e}")
    
    if not dataset_path:
        dataset_path = input("🔍 Enter CUB dataset path (e.g., data/CUB_200_2011): ")
    
    # Key files to check
    files_to_check = {
        'image_attribute_labels.txt': 4,  # image_id attribute_id is_present certainty
        'image_class_labels.txt': 2,      # image_id class_id
        'train_test_split.txt': 2,        # image_id is_training_image
        'images.txt': 2,                  # image_id image_path
        'classes.txt': 2                  # class_id class_name
    }
    
    all_good = True
    
    for filename, expected_fields in files_to_check.items():
        file_path = os.path.join(dataset_path, filename)
        if filename == 'image_attribute_labels.txt':
            file_path = os.path.join(dataset_path, 'attributes', filename)
        
        is_good = diagnose_file(file_path, expected_fields)
        
        if not is_good:
            all_good = False
            
            # Offer to fix the file
            answer = input(f"\n🔧 Would you like to fix {filename}? (y/n): ")
            if answer.lower() == 'y':
                backup_path = file_path + '.backup'
                fixed_path = file_path + '.fixed'
                
                # Create backup
                import shutil
                shutil.copy2(file_path, backup_path)
                print(f"📋 Backup created: {backup_path}")
                
                # Fix file
                fix_file(file_path, fixed_path, expected_fields)
                
                # Replace original
                replace = input(f"🔄 Replace original file with fixed version? (y/n): ")
                if replace.lower() == 'y':
                    shutil.move(fixed_path, file_path)
                    print(f"✅ File replaced: {file_path}")
                else:
                    print(f"💾 Fixed file saved as: {fixed_path}")
    
    if all_good:
        print("\n✅ All files look good!")
    else:
        print("\n⚠️  Some files have issues. Run the diagnostic again after fixing.")


if __name__ == "__main__":
    main()
