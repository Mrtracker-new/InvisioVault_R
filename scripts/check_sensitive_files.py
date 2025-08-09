#!/usr/bin/env python3
"""
Security Check Script for InvisioVault
=====================================

This script scans for potentially sensitive files that might accidentally
get committed to the repository despite .gitignore rules.

Usage:
    python scripts/check_sensitive_files.py

Author: Generated for InvisioVault Security
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set

# Patterns for sensitive files and content
SENSITIVE_FILE_PATTERNS = [
    r'\.key$', r'\.keyfile$', r'\.pem$', r'\.p12$', r'\.pfx$',
    r'\.env$', r'\.env\..*', r'\.secret$', r'\.credentials$',
    r'\.passwd$', r'\.password$', r'\.auth$', r'\.oauth$',
    r'\.token$', r'\.jwt$', r'config\.ini$', r'settings\.ini$',
    r'passwords\.txt$', r'secrets\.txt$', r'credentials\.txt$',
    r'service-account.*\.json$', r'client_secret.*\.json$',
    r'\.db$', r'\.sqlite.*$', r'\.log$'
]

SENSITIVE_CONTENT_PATTERNS = [
    r'password\s*=\s*["\'][^"\']{8,}["\']',
    r'secret\s*=\s*["\'][^"\']{16,}["\']',
    r'api_key\s*=\s*["\'][^"\']{16,}["\']',
    r'private_key\s*=\s*["\'][^"\']{32,}["\']',
    r'token\s*=\s*["\'][^"\']{16,}["\']',
    r'-----BEGIN [A-Z ]+-----',
    r'sk-[a-zA-Z0-9]{32,}',  # OpenAI-style API keys
    r'[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{27}',  # JWT pattern
]

EXCLUDED_DIRS = {
    '.git', '.venv', 'venv', '__pycache__', '.pytest_cache',
    '.mypy_cache', 'node_modules', '.idea', '.vscode',
    'temp', 'tmp', 'user_data', 'my_keys', 'test_outputs'
}

def scan_files(root_dir: Path) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Scan directory for sensitive files and content.
    
    Returns:
        Tuple of (sensitive_files, sensitive_content_matches)
    """
    sensitive_files = []
    sensitive_content = []
    
    # Compile patterns for efficiency
    file_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in SENSITIVE_FILE_PATTERNS]
    content_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in SENSITIVE_CONTENT_PATTERNS]
    
    for root, dirs, files in os.walk(root_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        
        for file in files:
            file_path = Path(root) / file
            relative_path = file_path.relative_to(root_dir)
            
            # Check file name patterns
            for pattern in file_patterns:
                if pattern.search(file):
                    sensitive_files.append(str(relative_path))
                    break
            
            # Check file content for text files
            if file_path.suffix in {'.py', '.txt', '.json', '.yml', '.yaml', '.ini', '.cfg', '.env'}:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    for pattern in content_patterns:
                        matches = pattern.findall(content)
                        if matches:
                            for match in matches:
                                sensitive_content.append((str(relative_path), match[:50] + '...'))
                            break
                            
                except (UnicodeDecodeError, PermissionError, OSError):
                    # Skip binary files or files we can't read
                    continue
    
    return sensitive_files, sensitive_content

def check_git_tracked_files(root_dir: Path) -> List[str]:
    """Check if any sensitive files are tracked by git."""
    import subprocess
    
    try:
        # Get list of tracked files
        result = subprocess.run(
            ['git', 'ls-files'], 
            capture_output=True, 
            text=True, 
            cwd=root_dir
        )
        
        if result.returncode != 0:
            return []
        
        tracked_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        # Check tracked files against sensitive patterns
        sensitive_tracked = []
        file_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in SENSITIVE_FILE_PATTERNS]
        
        for file_path in tracked_files:
            for pattern in file_patterns:
                if pattern.search(file_path):
                    sensitive_tracked.append(file_path)
                    break
        
        return sensitive_tracked
        
    except FileNotFoundError:
        print("⚠️  Git not found. Skipping git-tracked files check.")
        return []
    except Exception as e:
        print(f"⚠️  Error checking git-tracked files: {e}")
        return []

def main():
    """Main function to run security checks."""
    print("🔒 InvisioVault Security File Scanner")
    print("=" * 40)
    
    # Get project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print(f"📁 Scanning directory: {project_root}")
    print()
    
    # Scan for sensitive files
    sensitive_files, sensitive_content = scan_files(project_root)
    
    # Check git-tracked files
    git_tracked_sensitive = check_git_tracked_files(project_root)
    
    # Report results
    issues_found = 0
    
    if git_tracked_sensitive:
        issues_found += 1
        print("🚨 CRITICAL: Sensitive files tracked by Git!")
        for file_path in git_tracked_sensitive:
            print(f"   📄 {file_path}")
        print()
    
    if sensitive_files:
        issues_found += 1
        print("⚠️  Potentially sensitive files found:")
        for file_path in sensitive_files:
            print(f"   📄 {file_path}")
        print()
    
    if sensitive_content:
        issues_found += 1
        print("⚠️  Potentially sensitive content found:")
        for file_path, content in sensitive_content:
            print(f"   📄 {file_path}: {content}")
        print()
    
    if issues_found == 0:
        print("✅ No sensitive files or content detected!")
        print("✅ Your repository appears to be secure.")
    else:
        print(f"⚠️  Found {issues_found} potential security issue(s).")
        print("📋 Recommended actions:")
        print("   1. Review the files listed above")
        print("   2. Move sensitive files to appropriate locations")
        print("   3. Update .gitignore if needed")
        print("   4. Remove sensitive content from code")
        if git_tracked_sensitive:
            print("   5. Use 'git rm --cached <file>' to untrack sensitive files")
    
    print()
    print("🔍 Scan complete.")
    
    return 0 if issues_found == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
