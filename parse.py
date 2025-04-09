import json
import os
from pathlib import Path
from typing import Dict, List, Any

def validate_metadata(entry: Dict[str, Any]) -> List[str]:
    """
    Validate a single model entry's metadata.
    Returns a list of validation issues found.
    """
    issues = []
    
    # Check for library_name
    if 'library_name' not in entry:
        issues.append(f"Missing library_name for model {entry.get('id', 'unknown')}")
    
    # Check for pipeline_tag
    if 'pipeline_tag' not in entry:
        issues.append(f"Missing pipeline_tag for model {entry.get('id', 'unknown')}")
    
    # Check for license
    if 'license' not in entry:
        issues.append(f"Missing license for model {entry.get('id', 'unknown')}")
    
    return issues

def process_json_file(file_path: Path) -> Dict[str, List[str]]:
    """
    Process a single JSON file and return validation results.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    all_issues = []
    for entry in data:
        issues = validate_metadata(entry)
        if issues:
            all_issues.extend(issues)
    
    return {
        'file': str(file_path),
        'total_entries': len(data),
        'issues': all_issues
    }

def main():
    # Get all JSON files in the trending_data directory
    trending_dir = Path('trending_data')
    json_files = list(trending_dir.glob('*.json'))
    
    if not json_files:
        print("No JSON files found in trending_data directory")
        return
    
    # Process each file
    for json_file in json_files:
        print(f"\nProcessing {json_file.name}...")
        results = process_json_file(json_file)
        
        print(f"Total entries: {results['total_entries']}")
        if results['issues']:
            print("\nValidation Issues:")
            for issue in results['issues']:
                print(f"- {issue}")
        else:
            print("No validation issues found!")

if __name__ == "__main__":
    main() 