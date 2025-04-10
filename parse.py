import json
import os
from pathlib import Path
from typing import Dict, List, Any

def validate_metadata(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a single model entry's metadata.
    Returns a dictionary containing validation issues and model info.
    """
    issues = []
    model_id = entry.get('id', 'unknown')
    
    # Check for library_name
    if 'library_name' not in entry:
        issues.append("Missing library_name")
    
    # Check for pipeline_tag
    if 'pipeline_tag' not in entry:
        issues.append("Missing pipeline_tag")
    
    # Check for license
    if 'license' not in entry:
        issues.append("Missing license")
        
    # Check for config
    if 'config' not in entry:
        issues.append("Missing config")
    
    return {
        'model_id': model_id,
        'repo_url': f"https://huggingface.co/{model_id}",
        'issues': issues
    }

def process_json_file(file_path: Path) -> Dict[str, Any]:
    """
    Process a single JSON file and return validation results.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    model_issues = {}
    for entry in data:
        validation_result = validate_metadata(entry)
        if validation_result['issues']:
            model_issues[validation_result['model_id']] = validation_result
    
    return {
        'file': str(file_path),
        'total_entries': len(data),
        'models_with_issues': model_issues
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
        if results['models_with_issues']:
            print("\nModels with Validation Issues:")
            for model_id, model_data in results['models_with_issues'].items():
                print(f"\nModel: {model_id}")
                print(f"Repository: {model_data['repo_url']}")
                print("Issues:")
                for issue in model_data['issues']:
                    print(f"  - {issue}")
        else:
            print("No validation issues found!")

if __name__ == "__main__":
    main() 