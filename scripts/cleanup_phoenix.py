#!/usr/bin/env python3
"""
Phoenix Cleanup Script for the Contextual RAG Chatbot.

This script helps clean up Phoenix temporary files that might be causing
file access errors on Windows systems.
"""

import os
import shutil
import tempfile
import logging
from pathlib import Path
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_phoenix_temp_files():
    """
    Find Phoenix temporary files that might be locked.
    
    Returns:
        List[Path]: List of Phoenix temp file paths
    """
    phoenix_files = []
    
    # Common temp directories
    temp_dirs = [
        tempfile.gettempdir(),
        os.path.expanduser("~/AppData/Local/Temp"),  # Windows
        "/tmp",  # Linux/Mac
    ]
    
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            logger.info(f"Searching in: {temp_dir}")
            
            # Look for Phoenix-related files
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if 'phoenix' in file.lower() or 'phoenix.db' in file:
                        file_path = Path(root) / file
                        phoenix_files.append(file_path)
                        logger.info(f"Found Phoenix file: {file_path}")
    
    return phoenix_files

def kill_phoenix_processes():
    """
    Kill any running Phoenix processes.
    
    Returns:
        int: Number of processes killed
    """
    killed_count = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if process is related to Phoenix
            if any(keyword in proc.info['name'].lower() for keyword in ['phoenix', 'python']):
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'phoenix' in cmdline.lower():
                    logger.info(f"Killing Phoenix process: {proc.info['name']} (PID: {proc.info['pid']})")
                    proc.kill()
                    killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return killed_count

def cleanup_phoenix_files():
    """
    Clean up Phoenix temporary files.
    
    Returns:
        Dict[str, Any]: Cleanup results
    """
    results = {
        'files_found': 0,
        'files_removed': 0,
        'files_locked': 0,
        'processes_killed': 0,
        'errors': []
    }
    
    try:
        # First, kill any Phoenix processes
        logger.info("Checking for Phoenix processes...")
        results['processes_killed'] = kill_phoenix_processes()
        
        if results['processes_killed'] > 0:
            logger.info(f"Killed {results['processes_killed']} Phoenix processes")
        
        # Find Phoenix temp files
        logger.info("Searching for Phoenix temporary files...")
        phoenix_files = find_phoenix_temp_files()
        results['files_found'] = len(phoenix_files)
        
        if not phoenix_files:
            logger.info("No Phoenix temporary files found")
            return results
        
        # Try to remove files
        for file_path in phoenix_files:
            try:
                if file_path.exists():
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                    
                    logger.info(f"Removed: {file_path}")
                    results['files_removed'] += 1
                    
            except PermissionError:
                logger.warning(f"File locked (cannot remove): {file_path}")
                results['files_locked'] += 1
            except Exception as e:
                logger.error(f"Error removing {file_path}: {e}")
                results['errors'].append(str(e))
        
        logger.info(f"Cleanup completed: {results['files_removed']} files removed, {results['files_locked']} files locked")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        results['errors'].append(str(e))
    
    return results

def main():
    """Main function to run Phoenix cleanup."""
    logger.info("🧹 Starting Phoenix cleanup...")
    
    results = cleanup_phoenix_files()
    
    logger.info("\n📊 Cleanup Results:")
    logger.info(f"  - Files found: {results['files_found']}")
    logger.info(f"  - Files removed: {results['files_removed']}")
    logger.info(f"  - Files locked: {results['files_locked']}")
    logger.info(f"  - Processes killed: {results['processes_killed']}")
    
    if results['errors']:
        logger.warning(f"  - Errors: {len(results['errors'])}")
        for error in results['errors']:
            logger.warning(f"    - {error}")
    
    if results['files_locked'] > 0:
        logger.warning("\n⚠️  Some files are still locked. You may need to:")
        logger.warning("  1. Restart your computer")
        logger.warning("  2. Close any applications using Phoenix")
        logger.warning("  3. Run this script again")
    
    logger.info("\n✅ Phoenix cleanup completed!")

if __name__ == "__main__":
    main() 