#!/usr/bin/env python3
"""
Security Check Script for InVisioVault
Author: Rolan (RNR)
Purpose: Automated security verification and sensitive data detection
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Set

# Sensitive file patterns to search for
SENSITIVE_PATTERNS = {
    'keyfiles': ['*.key', '*.keyfile', '*.pem', '*.p12', '*.pfx'],
    'passwords': ['*password*', '*passwd*', '*credentials*'],
    'environment': ['*.env', '.env*', '*secret*'],
    'config': ['config.ini', 'settings.ini', '*.cfg'],
    'databases': ['*.db', '*.sqlite*'],
    'steganographic': ['*.stego', '*.hidden', '*encrypted_img*'],
}

# Sensitive content patterns in files
CONTENT_PATTERNS = [
    r'password\s*=\s*["\'][^"\']{4,}["\']',  # Only match passwords with 4+ chars
    r'api[_-]?key\s*=\s*["\'][a-zA-Z0-9_-]{10,}["\']',  # Only match realistic API keys
    r'secret\s*=\s*["\'][a-zA-Z0-9_-]{8,}["\']',  # Only match realistic secrets
    r'token\s*=\s*["\'][a-zA-Z0-9_-]{16,}["\']',  # Only match realistic tokens
    r'BEGIN\s+(PRIVATE\s+KEY|RSA\s+PRIVATE\s+KEY)',
]

# Whitelisted files that legitimately contain security-related terms
WHITELISTED_FILES = {
    'ui/components/password_input.py',
    'utils/password_validator.py', 
    'core/encryption_engine.py',
    'core/steganography_engine.py',
    'core/security_manager.py',
    'SECURITY.md',
    'README.md',
    'scripts/security_check.py'
}

# Test files that may contain hardcoded test data
TEST_FILES = {
    'test_main.py',
    'demo_performance.py'
}

def find_sensitive_files(project_root: Path) -> Dict[str, List[Path]]:
    """Find files that match sensitive patterns."""
    found_files = {}
    
    # Directories to skip completely
    skip_dirs = {'.git', '.venv', 'venv', 'env', '__pycache__', 'node_modules'}
    
    for category, patterns in SENSITIVE_PATTERNS.items():
        found_files[category] = []
        for pattern in patterns:
            for file_path in project_root.rglob(pattern):
                # Skip files in ignored directories
                if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                    continue
                
                # Skip legitimate project files that happen to match patterns
                relative_path = str(file_path.relative_to(project_root)).replace('\\', '/')
                if relative_path in WHITELISTED_FILES:
                    continue
                    
                found_files[category].append(file_path)
    
    return found_files

def scan_file_contents(file_path: Path) -> List[str]:
    """Scan file contents for sensitive patterns."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        matches = []
        for pattern in CONTENT_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                matches.append(pattern)
        
        return matches
    except Exception:
        return []

def check_git_status() -> Dict[str, List[str]]:
    """Check git status for untracked sensitive files."""
    import subprocess
    
    try:
        # Get untracked files
        result = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard'], 
                              capture_output=True, text=True)
        untracked = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        # Get tracked files
        result = subprocess.run(['git', 'ls-files'], 
                              capture_output=True, text=True)
        tracked = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        return {'untracked': untracked, 'tracked': tracked}
    except Exception as e:
        print(f"Warning: Could not check git status: {e}")
        return {'untracked': [], 'tracked': []}

def main():
    """Main security check function."""
    project_root = Path(__file__).parent.parent
    print("🔒 InVisioVault Security Check")
    print("=" * 50)
    print(f"Scanning: {project_root}")
    print()
    
    # Find sensitive files
    print("🔍 Scanning for sensitive files...")
    sensitive_files = find_sensitive_files(project_root)
    
    total_sensitive = sum(len(files) for files in sensitive_files.values())
    if total_sensitive > 0:
        print(f"⚠️  Found {total_sensitive} potentially sensitive files:")
        for category, files in sensitive_files.items():
            if files:
                print(f"  {category.upper()}:")
                for file_path in files:
                    print(f"    - {file_path.relative_to(project_root)}")
        print()
    else:
        print("✅ No sensitive files found")
        print()
    
    # Check git status
    print("🔍 Checking git status...")
    git_status = check_git_status()
    
    # Check if sensitive files are tracked
    tracked_sensitive = []
    for file_path in git_status['tracked']:
        path_obj = project_root / file_path
        if any(path_obj.match(pattern) for patterns in SENSITIVE_PATTERNS.values() for pattern in patterns):
            tracked_sensitive.append(file_path)
    
    if tracked_sensitive:
        print(f"🚨 CRITICAL: {len(tracked_sensitive)} sensitive files are being tracked by git:")
        for file_path in tracked_sensitive:
            print(f"    - {file_path}")
        print("  ACTION REQUIRED: Remove these files from git tracking!")
        print()
    else:
        print("✅ No sensitive files are being tracked by git")
        print()
    
    # Scan Python files for hardcoded secrets
    print("🔍 Scanning Python files for hardcoded secrets...")
    python_files = list(project_root.rglob("*.py"))
    files_with_secrets = []
    whitelisted_files = []
    test_files_with_secrets = []
    
    for py_file in python_files:
        if '.venv' not in py_file.parts and '__pycache__' not in py_file.parts:
            matches = scan_file_contents(py_file)
            if matches:
                relative_path = str(py_file.relative_to(project_root)).replace('\\', '/')
                
                if relative_path in WHITELISTED_FILES:
                    whitelisted_files.append((py_file, matches))
                elif relative_path in TEST_FILES:
                    test_files_with_secrets.append((py_file, matches))
                else:
                    files_with_secrets.append((py_file, matches))
    
    # Report critical secrets (non-whitelisted, non-test files)
    if files_with_secrets:
        print(f"🚨 CRITICAL: Found potential secrets in {len(files_with_secrets)} Python files:")
        for file_path, patterns in files_with_secrets:
            print(f"  - {file_path.relative_to(project_root)}")
            for pattern in patterns:
                print(f"    Pattern: {pattern}")
        print()
    else:
        print("✅ No critical hardcoded secrets found in Python files")
    
    # Report test file secrets (warnings)
    if test_files_with_secrets:
        print(f"⚠️  Found test credentials in {len(test_files_with_secrets)} test files (review recommended):")
        for file_path, patterns in test_files_with_secrets:
            print(f"  - {file_path.relative_to(project_root)}")
            for pattern in patterns:
                print(f"    Pattern: {pattern}")
        print("  💡 Consider using environment variables or mock data in tests")
        print()
    
    # Report whitelisted files (informational)
    if whitelisted_files:
        print(f"ℹ️  Found security terms in {len(whitelisted_files)} whitelisted files (expected):")
        for file_path, patterns in whitelisted_files:
            print(f"  - {file_path.relative_to(project_root)} ({len(patterns)} patterns)")
        print()
    
    print()
    
    # Check .gitignore effectiveness
    print("🔍 Checking .gitignore effectiveness...")
    gitignore_path = project_root / ".gitignore"
    
    if not gitignore_path.exists():
        print("🚨 CRITICAL: No .gitignore file found!")
    else:
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read()
        
        # Check for key patterns
        required_patterns = ['*.key', '*.env', 'user_data/', 'my_keys/', '__pycache__/']
        missing_patterns = []
        
        for pattern in required_patterns:
            if pattern not in gitignore_content:
                missing_patterns.append(pattern)
        
        if missing_patterns:
            print(f"⚠️  Missing {len(missing_patterns)} important patterns in .gitignore:")
            for pattern in missing_patterns:
                print(f"    - {pattern}")
        else:
            print("✅ .gitignore contains essential security patterns")
        print()
    
    # Summary
    print("📋 SECURITY SUMMARY")
    print("-" * 30)
    
    security_score = 0
    total_checks = 4
    
    if total_sensitive == 0:
        security_score += 1
        print("✅ No sensitive files found")
    else:
        print(f"⚠️  {total_sensitive} sensitive files found")
    
    if not tracked_sensitive:
        security_score += 1
        print("✅ No sensitive files tracked by git")
    else:
        print(f"🚨 {len(tracked_sensitive)} sensitive files tracked by git")
    
    if not files_with_secrets:
        security_score += 1
        print("✅ No hardcoded secrets in Python files")
    else:
        print(f"⚠️  {len(files_with_secrets)} files with potential secrets")
    
    if gitignore_path.exists() and not missing_patterns:
        security_score += 1
        print("✅ .gitignore properly configured")
    else:
        print("⚠️  .gitignore needs improvement")
    
    print()
    print(f"🎯 Security Score: {security_score}/{total_checks}")
    
    if security_score == total_checks:
        print("🎉 Excellent! Your project is secure.")
        return 0
    elif security_score >= total_checks * 0.75:
        print("✅ Good security posture with minor issues.")
        return 0
    else:
        print("⚠️  Security issues found. Please review and fix.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
