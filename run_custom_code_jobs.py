import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import datetime
import logging
from dotenv import load_dotenv

import requests
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, list_repo_files, upload_file

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Configuration & Constants ===

TARGET_MODEL_IDS = [
    "openbmb/MiniCPM4-8B",
    "inclusionAI/Ming-Lite-Omni", 
    "deepseek-ai/DeepSeek-R1-0528",
    "Motif-Technologies/Motif-2.6B",
    "openbmb/MiniCPM4-0.5B",
    "deepseek-ai/DeepSeek-R1",
]

HF_DATASET_CODE_REPO = "model-metadata/custom-code-py-files"
HF_DATASET_VRAM_REPO = "model-metadata/custom-vram-code"
HF_DATASET_EXCEPTIONS_REPO = "model-metadata/model-code-exception"

LOCAL_CODE_DIR = Path("custom_code")
LOCAL_CODE_DIR.mkdir(parents=True, exist_ok=True)

# GPU configurations with VRAM in GB
GPU_VRAM_MAPPING = {
    "a10g-small": 15,
    "l4x1": 30,
    "a10g-large": 46,
}

# Reverse mapping for VRAM to GPU selection
VRAM_TO_GPU_MAPPING = dict(sorted({vram: gpu for gpu, vram in GPU_VRAM_MAPPING.items()}.items()))

DOCKER_IMAGE_URL = "ghcr.io/astral-sh/uv:debian"
# DOCKER_IMAGE_URL = "aritrarg/uv:debian-chown"

# UV script header for Python dependencies
UV_SCRIPT_HEADER = """\
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "transformers",
#     "torch",
# ]
# ///
"""

# === Helper Functions ===

def get_current_timestamp() -> str:
    """Get current date in DD-MM-YY format."""
    return datetime.date.today().strftime("%d-%m-%y")

def sanitize_model_name(model_id: str) -> str:
    """Convert model ID to filesystem-safe name."""
    return "_".join(model_id.split("/"))

def check_existing_files(repo_id: str, repo_type: str = "dataset") -> set:
    """Get list of already uploaded files in the repository."""
    try:
        files = list_repo_files(repo_id, repo_type=repo_type)
        return set(files)
    except Exception as e:
        logger.warning(f"Failed to list files in {repo_id}: {e}")
        return set()

# === Notebook Processing ===

def fetch_notebook_content(model_id: str) -> Optional[Dict]:
    """
    Fetch Jupyter notebook from HuggingFace model repository.
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        Parsed notebook content or None if fetch fails
    """
    notebook_url = f"https://huggingface.co/{model_id}.ipynb"
    
    try:
        response = requests.get(notebook_url, timeout=30)
        response.raise_for_status()
        return json.loads(response.text)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch notebook for {model_id}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in notebook for {model_id}: {e}")
        return None

def extract_code_cells(notebook: Dict) -> List[str]:
    """Extract code cells from Jupyter notebook that start with #."""
    code_snippets = []
    
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
            
        source_lines = cell.get("source", [])
        cell_content = "".join(source_lines).strip()
        
        # Only process cells that start with # (likely examples or documentation)
        if cell_content and cell_content.startswith("#"):
            code_snippets.append(cell_content)
    
    return code_snippets

def wrap_code_with_error_handling(code_content: str, model_name: str, snippet_index: int) -> str:
    """
    Wrap code snippet with try-except block for error logging.
    
    Args:
        code_content: Original code snippet
        model_name: Sanitized model name
        snippet_index: Index of the snippet
        
    Returns:
        Code wrapped with error handling
    """
    exception_filename = f"{model_name}_{snippet_index}_exception.txt"
    timestamp = get_current_timestamp()
    
    wrapped_lines = ["try:"]
    
    # Indent original code
    for line in code_content.splitlines():
        wrapped_lines.append(f"    {line}")
    
    # Add exception handling
    wrapped_lines.extend([
        "except Exception as e:",
        f"    exception_file = '{exception_filename}'",
        "    with open(exception_file, 'w') as f:",
        "        import traceback",
        "        traceback.print_exc(file=f)",
        "    # Upload exception log to HuggingFace",
        "    from huggingface_hub import upload_file",
        "    upload_file(",
        f"        path_or_fileobj=exception_file,",
        f"        repo_id='{HF_DATASET_EXCEPTIONS_REPO}',",
        f"        path_in_repo='{timestamp}/{exception_filename}',",
        "        repo_type='dataset',",
        "    )"
    ])
    
    return "\n".join(wrapped_lines)

def process_notebook_to_scripts(model_id: str) -> List[Tuple[str, str]]:
    """
    Convert notebook code cells to executable Python scripts.
    
    Returns:
        List of (filename, script_content) tuples
    """
    notebook = fetch_notebook_content(model_id)
    if not notebook:
        return []
    
    code_snippets = extract_code_cells(notebook)
    if not code_snippets:
        logger.info(f"No code cells found in notebook for {model_id}")
        return []
    
    timestamp = get_current_timestamp()
    model_name = sanitize_model_name(model_id)
    processed_scripts = []
    
    for idx, snippet in enumerate(code_snippets):
        wrapped_code = wrap_code_with_error_handling(snippet, model_name, idx)
        full_script = UV_SCRIPT_HEADER + "\n" + wrapped_code
        
        script_filename = f"{timestamp}/{model_name}_{idx}.py"
        processed_scripts.append((script_filename, full_script))
    
    logger.info(f"Processed {len(processed_scripts)} scripts for {model_id}")
    return processed_scripts

# === File Upload Management ===

def upload_scripts_to_dataset(model_id: str) -> List[str]:
    """
    Process and upload notebook scripts to HuggingFace dataset.
    
    Args:
        model_id: HuggingFace model identifier
        existing_files: Set of already uploaded file paths
        
    Returns:
        List of successfully uploaded file paths
    """
    scripts = process_notebook_to_scripts(model_id)
    if not scripts:
        return []
    
    uploaded_paths = []
    
    for filename, script_content in scripts:
        remote_path = f"custom_code/{filename}"
        
        # Write locally first
        local_filename = filename.split("/")[-1]
        local_path = LOCAL_CODE_DIR / local_filename
        local_path.write_text(script_content, encoding="utf-8")
        
        try:
            upload_file(
                repo_id=HF_DATASET_CODE_REPO,
                path_or_fileobj=local_path,
                path_in_repo=remote_path,
                repo_type="dataset",
            )
            logger.info(f"Successfully uploaded: {remote_path}")
            uploaded_paths.append(remote_path)
        except Exception as e:
            logger.error(f"Failed to upload {remote_path}: {e}")
        finally:
            # Clean up local file
            if local_path.exists():
                local_path.unlink()
    
    return uploaded_paths

# === VRAM Estimation ===

def estimate_model_vram(model_id: str) -> float:
    """
    Estimate VRAM usage based on model parameters.
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        Estimated VRAM in GB
    """
    try:
        api = HfApi()
        model_info = api.model_info(model_id)
        
        if model_info.safetensors is None:
            logger.warning(f"No safetensors info for {model_id}")
            return 0.0
        
        total_params = model_info.safetensors.total
        param_dtypes = list(model_info.safetensors.parameters.keys())
        
        # Determine bytes per parameter based on dtype
        primary_dtype = param_dtypes[0] if param_dtypes else "FP32"
        bytes_per_param = 2 if "BF16" in primary_dtype or "FP16" in primary_dtype else 4
        
        # Calculate VRAM in GB
        vram_gb = (total_params * bytes_per_param) / (1024**3)
        
        # Add overhead for inference (typically 20-30%)
        vram_with_overhead = vram_gb * 1.3
        
        return round(vram_with_overhead, 2)
        
    except Exception as e:
        logger.error(f"Failed to estimate VRAM for {model_id}: {e}")
        return 0.0

def select_appropriate_gpu(vram_required: float) -> Optional[str]:
    """
    Select the smallest GPU that can handle the required VRAM.
    
    Args:
        vram_required: Required VRAM in GB
        
    Returns:
        GPU identifier or None if no suitable GPU
    """
    for available_vram, gpu_type in VRAM_TO_GPU_MAPPING.items():
        if vram_required < available_vram * 0.9:  # Leave 10% headroom
            return gpu_type
    
    logger.warning(f"No suitable GPU found for {vram_required:.2f} GB VRAM requirement")
    return None

# === Dataset Management ===

def update_vram_dataset(vram_records: List[Dict[str, any]]):
    """Upload VRAM metadata to HuggingFace dataset."""
    if not vram_records:
        logger.warning("No VRAM records to upload")
        return
    
    try:
        dataset = Dataset.from_list(vram_records)
        dataset.push_to_hub(HF_DATASET_VRAM_REPO, split="train")
        logger.info(f"Successfully updated VRAM dataset with {len(vram_records)} records")
    except Exception as e:
        logger.error(f"Failed to update VRAM dataset: {e}")

# === Job Execution ===

def launch_hf_jobs():
    """Launch jobs using HuggingFace's hfjobs CLI for uploaded scripts."""
    try:
        # Load VRAM dataset
        vram_dataset = load_dataset(HF_DATASET_VRAM_REPO)["train"]
        
        # Get list of uploaded scripts
        uploaded_scripts = check_existing_files(HF_DATASET_CODE_REPO)
        
        for record in vram_dataset:
            model_id = record["model_id"]
            required_vram = record["vram"]
            model_name = sanitize_model_name(model_id)
            
            # Find matching scripts
            matching_scripts = [
                script for script in uploaded_scripts
                if script.startswith("custom_code/") and 
                   script.endswith(".py") and 
                   model_name in script
            ]
            
            if not matching_scripts:
                logger.warning(f"No scripts found for {model_id}")
                continue
            
            # Select appropriate GPU
            selected_gpu = select_appropriate_gpu(required_vram)
            if not selected_gpu:
                logger.warning(f"Skipping {model_id}: VRAM requirement too high ({required_vram:.2f} GB)")
                continue
            
            # Launch job for each script
            for script_path in matching_scripts:
                script_url = f"https://huggingface.co/datasets/{HF_DATASET_CODE_REPO}/raw/main/{script_path}"
                
                # launch_command = (
                #     f"hfjobs run --detach --flavor {selected_gpu} {DOCKER_IMAGE_URL} /bin/bash -c "
                #     f'"export HOME=/tmp && export USER=dummy && uv run {script_url}"'
                # )
                launch_command = (
                    f"hfjobs run --detach --secret HF_TOKEN={os.getenv('HF_TOKEN')} --flavor {selected_gpu} {DOCKER_IMAGE_URL} /bin/bash -c "
                    f'"export HOME=/tmp && export USER=dummy && uv run {script_url}"'
                )
                # launch_command = (
                #     f"hfjobs run --detach --flavor {selected_gpu} {DOCKER_IMAGE_URL} /bin/bash -c "
                #     f'"chown -R 1000:1000 /tmp && export HOME=/tmp && export USER=dummy && uv run {script_url}"'
                # )

                
                try:
                    exit_code = os.system(launch_command)
                    if exit_code == 0:
                        logger.info(f"Successfully launched job for {model_id} on {selected_gpu}")
                    else:
                        logger.error(f"Failed to launch job for {model_id}, exit code: {exit_code}")
                except Exception as e:
                    logger.error(f"Error launching job for {model_id}: {e}")
                    
    except Exception as e:
        logger.error(f"Failed to launch jobs: {e}")

# === Main Controller ===

def main():
    """Main execution flow."""
    logger.info("Starting HuggingFace model processing pipeline")
    
    vram_metadata_records = []
    successful_models = 0
    
    for model_id in TARGET_MODEL_IDS:
        logger.info(f"Processing model: {model_id}")
        
        # Upload scripts
        uploaded_paths = upload_scripts_to_dataset(model_id)
        
        if not uploaded_paths:
            logger.warning(f"No scripts uploaded for {model_id}")
            continue
        
        # Estimate VRAM
        estimated_vram = estimate_model_vram(model_id)
        if estimated_vram > 0:
            vram_metadata_records.append({
                "model_id": model_id,
                "vram": estimated_vram,
                "timestamp": get_current_timestamp()
            })
            logger.info(f"Estimated VRAM for {model_id}: {estimated_vram:.2f} GB")
            successful_models += 1

    
    # Update VRAM dataset
    if vram_metadata_records:
        update_vram_dataset(vram_metadata_records)
    
    # Launch jobs
    if successful_models > 0:
        logger.info("Launching HuggingFace jobs...")
        launch_hf_jobs()
    else:
        logger.warning("No models were successfully processed")
    
    logger.info("Pipeline execution completed")

# === Entry Point ===

if __name__ == "__main__":
    main()
