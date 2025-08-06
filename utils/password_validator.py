"""
Password Validation System
Comprehensive password validation with strength scoring and requirements.
"""

import re
import string
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

from utils.logger import Logger
from utils.config_manager import ConfigManager, ConfigSection


class PasswordStrength(Enum):
    """Password strength levels."""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class PasswordValidator:
    """Comprehensive password validation system."""
    
    # Common weak passwords and patterns
    COMMON_PASSWORDS = {
        'password', 'password123', '123456', '123456789', 'qwerty', 'abc123',
        'password1', 'admin', 'letmein', 'welcome', 'monkey', 'dragon',
        'master', 'shadow', 'trustno1', 'freedom', '111111', '000000'
    }
    
    # Common keyboard patterns
    KEYBOARD_PATTERNS = {
        'qwertyuiop', 'asdfghjkl', 'zxcvbnm', '1234567890',
        'qwerty', 'asdfgh', 'zxcvbn', '123456', '654321'
    }
    
    def __init__(self, config: Optional[ConfigManager] = None):
        self.logger = Logger()
        self.config = config or ConfigManager()
        
        # Load configuration
        self.min_length = self.config.get(ConfigSection.SECURITY, "password_min_length", 8)
        self.require_uppercase = self.config.get(ConfigSection.SECURITY, "password_require_uppercase", True)
        self.require_lowercase = self.config.get(ConfigSection.SECURITY, "password_require_lowercase", True)
        self.require_numbers = self.config.get(ConfigSection.SECURITY, "password_require_numbers", True)
        self.require_symbols = self.config.get(ConfigSection.SECURITY, "password_require_symbols", True)
        
        self.logger.debug("Password validator initialized")
    
    def validate_password(self, password: str) -> Dict[str, Any]:
        """Validate password against all requirements.
        
        Args:
            password: Password to validate
            
        Returns:
            Validation result dictionary
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'strength': PasswordStrength.WEAK,
            'score': 0,
            'requirements': {
                'min_length': False,
                'uppercase': False,
                'lowercase': False,
                'numbers': False,
                'symbols': False
            },
            'entropy_bits': 0.0
        }
        
        try:
            # Check basic requirements
            self._check_length(password, result)
            self._check_character_types(password, result)
            
            # Check for common weaknesses
            self._check_common_passwords(password, result)
            self._check_patterns(password, result)
            self._check_repetition(password, result)
            
            # Calculate strength and entropy
            result['score'] = self._calculate_score(password, result)
            result['strength'] = self._determine_strength(result['score'])
            result['entropy_bits'] = self._calculate_entropy(password)
            
            # Overall validation
            result['valid'] = len(result['errors']) == 0
            
            self.logger.debug(f"Password validation: score={result['score']}, strength={result['strength'].value}")
            
        except Exception as e:
            self.logger.error(f"Error validating password: {e}")
            result['valid'] = False
            result['errors'].append("Validation error occurred")
        
        return result
    
    def _check_length(self, password: str, result: Dict) -> None:
        """Check password length requirement."""
        if len(password) >= self.min_length:
            result['requirements']['min_length'] = True
        else:
            result['errors'].append(f"Password must be at least {self.min_length} characters long")
    
    def _check_character_types(self, password: str, result: Dict) -> None:
        """Check character type requirements."""
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_symbol = any(c in string.punctuation for c in password)
        
        result['requirements']['uppercase'] = has_upper
        result['requirements']['lowercase'] = has_lower
        result['requirements']['numbers'] = has_digit
        result['requirements']['symbols'] = has_symbol
        
        if self.require_uppercase and not has_upper:
            result['errors'].append("Password must contain at least one uppercase letter")
        
        if self.require_lowercase and not has_lower:
            result['errors'].append("Password must contain at least one lowercase letter")
        
        if self.require_numbers and not has_digit:
            result['errors'].append("Password must contain at least one number")
        
        if self.require_symbols and not has_symbol:
            result['errors'].append("Password must contain at least one symbol")
    
    def _check_common_passwords(self, password: str, result: Dict) -> None:
        """Check for common weak passwords."""
        lower_password = password.lower()
        
        if lower_password in self.COMMON_PASSWORDS:
            result['errors'].append("Password is too common and easily guessable")
        
        # Check for common patterns with numbers
        base_patterns = ['password', 'admin', 'user', 'login']
        for pattern in base_patterns:
            if lower_password.startswith(pattern) and len(lower_password) <= len(pattern) + 3:
                result['warnings'].append(f"Avoid using '{pattern}' as a base for passwords")
    
    def _check_patterns(self, password: str, result: Dict) -> None:
        """Check for keyboard patterns and sequences."""
        lower_password = password.lower()
        
        # Check keyboard patterns
        for pattern in self.KEYBOARD_PATTERNS:
            if pattern in lower_password or pattern[::-1] in lower_password:
                result['warnings'].append("Avoid keyboard patterns like 'qwerty' or '123456'")
                break
        
        # Check for sequential characters
        if self._has_sequential_chars(password):
            result['warnings'].append("Avoid sequential characters like 'abc' or '123'")
    
    def _check_repetition(self, password: str, result: Dict) -> None:
        """Check for excessive character repetition."""
        if len(password) == 0:
            return
        
        # Check for repeated characters
        max_repeat = 1
        current_repeat = 1
        
        for i in range(1, len(password)):
            if password[i] == password[i-1]:
                current_repeat += 1
                max_repeat = max(max_repeat, current_repeat)
            else:
                current_repeat = 1
        
        if max_repeat >= 3:
            result['warnings'].append("Avoid repeating the same character multiple times")
        
        # Check for repeated patterns
        if self._has_repeated_patterns(password):
            result['warnings'].append("Avoid repeating patterns like 'abcabc' or '123123'")
    
    def _has_sequential_chars(self, password: str, min_length: int = 3) -> bool:
        """Check if password contains sequential characters."""
        sequences = [
            string.ascii_lowercase,
            string.ascii_uppercase,
            string.digits
        ]
        
        for sequence in sequences:
            for i in range(len(sequence) - min_length + 1):
                subseq = sequence[i:i + min_length]
                if subseq in password or subseq[::-1] in password:
                    return True
        
        return False
    
    def _has_repeated_patterns(self, password: str, min_pattern_length: int = 2) -> bool:
        """Check if password has repeated patterns."""
        for pattern_length in range(min_pattern_length, len(password) // 2 + 1):
            for i in range(len(password) - pattern_length * 2 + 1):
                pattern = password[i:i + pattern_length]
                rest = password[i + pattern_length:]
                if rest.startswith(pattern):
                    return True
        
        return False
    
    def _calculate_score(self, password: str, result: Dict) -> int:
        """Calculate password strength score (0-100)."""
        score = 0
        
        # Length bonus (up to 25 points)
        if len(password) >= 12:
            score += 25
        elif len(password) >= 8:
            score += 15
        elif len(password) >= 6:
            score += 10
        else:
            score += max(0, len(password) * 2)
        
        # Character diversity (up to 40 points)
        char_types = 0
        if result['requirements']['lowercase']:
            char_types += 1
            score += 5
        if result['requirements']['uppercase']:
            char_types += 1
            score += 5
        if result['requirements']['numbers']:
            char_types += 1
            score += 10
        if result['requirements']['symbols']:
            char_types += 1
            score += 20
        
        # Bonus for using all character types
        if char_types == 4:
            score += 10
        
        # Unique characters bonus (up to 15 points)
        unique_chars = len(set(password))
        uniqueness_ratio = unique_chars / len(password) if password else 0
        score += int(uniqueness_ratio * 15)
        
        # Entropy bonus (up to 20 points)
        entropy = self._calculate_entropy(password)
        if entropy >= 60:
            score += 20
        elif entropy >= 40:
            score += 15
        elif entropy >= 25:
            score += 10
        elif entropy >= 15:
            score += 5
        
        # Penalties
        score -= len(result['errors']) * 10
        score -= len(result['warnings']) * 5
        
        return max(0, min(100, score))
    
    def _determine_strength(self, score: int) -> PasswordStrength:
        """Determine password strength based on score."""
        if score >= 80:
            return PasswordStrength.VERY_STRONG
        elif score >= 60:
            return PasswordStrength.STRONG
        elif score >= 40:
            return PasswordStrength.MODERATE
        elif score >= 20:
            return PasswordStrength.WEAK
        else:
            return PasswordStrength.VERY_WEAK
    
    def _calculate_entropy(self, password: str) -> float:
        """Calculate password entropy in bits."""
        if not password:
            return 0.0
        
        # Determine character set size
        charset_size = 0
        
        if any(c.islower() for c in password):
            charset_size += 26  # lowercase letters
        if any(c.isupper() for c in password):
            charset_size += 26  # uppercase letters
        if any(c.isdigit() for c in password):
            charset_size += 10  # digits
        if any(c in string.punctuation for c in password):
            charset_size += len(string.punctuation)  # symbols
        
        if charset_size == 0:
            return 0.0
        
        # Calculate entropy: log2(charset_size^length)
        import math
        entropy = len(password) * math.log2(charset_size)
        
        return entropy
    
    def generate_suggestions(self, password: str) -> List[str]:
        """Generate suggestions to improve password strength.
        
        Args:
            password: Current password
            
        Returns:
            List of improvement suggestions
        """
        validation = self.validate_password(password)
        suggestions = []
        
        if not validation['requirements']['min_length']:
            suggestions.append(f"Make your password at least {self.min_length} characters long")
        
        if not validation['requirements']['uppercase']:
            suggestions.append("Add uppercase letters (A-Z)")
        
        if not validation['requirements']['lowercase']:
            suggestions.append("Add lowercase letters (a-z)")
        
        if not validation['requirements']['numbers']:
            suggestions.append("Add numbers (0-9)")
        
        if not validation['requirements']['symbols']:
            suggestions.append("Add symbols (!@#$%^&*)")
        
        if validation['strength'] in [PasswordStrength.VERY_WEAK, PasswordStrength.WEAK]:
            suggestions.extend([
                "Use a longer password (12+ characters recommended)",
                "Avoid common words and patterns",
                "Use a mix of unrelated words or a passphrase",
                "Consider using a password manager"
            ])
        
        if validation['entropy_bits'] < 50:
            suggestions.append("Increase complexity to improve entropy")
        
        return suggestions
    
    def get_strength_description(self, strength: PasswordStrength) -> str:
        """Get human-readable description of password strength."""
        descriptions = {
            PasswordStrength.VERY_WEAK: "Very Weak - Easily cracked",
            PasswordStrength.WEAK: "Weak - May be cracked quickly",
            PasswordStrength.MODERATE: "Moderate - Reasonable protection",
            PasswordStrength.STRONG: "Strong - Good protection",
            PasswordStrength.VERY_STRONG: "Very Strong - Excellent protection"
        }
        return descriptions.get(strength, "Unknown")
    
    def estimate_crack_time(self, password: str) -> str:
        """Estimate time to crack password with brute force.
        
        Args:
            password: Password to analyze
            
        Returns:
            Human-readable crack time estimate
        """
        entropy = self._calculate_entropy(password)
        
        # Assume 1 billion attempts per second (modern hardware)
        attempts_per_second = 1e9
        
        # Average case: need to try half the keyspace
        total_combinations = 2 ** (entropy - 1)
        seconds_to_crack = total_combinations / attempts_per_second
        
        # Convert to human readable format
        if seconds_to_crack < 1:
            return "Instantly"
        elif seconds_to_crack < 60:
            return f"{seconds_to_crack:.0f} seconds"
        elif seconds_to_crack < 3600:
            return f"{seconds_to_crack/60:.0f} minutes"
        elif seconds_to_crack < 86400:
            return f"{seconds_to_crack/3600:.0f} hours"
        elif seconds_to_crack < 31536000:
            return f"{seconds_to_crack/86400:.0f} days"
        elif seconds_to_crack < 31536000000:
            return f"{seconds_to_crack/31536000:.0f} years"
        else:
            return "Centuries+"
