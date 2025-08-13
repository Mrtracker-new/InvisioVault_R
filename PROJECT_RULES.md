# InVisioVault Project Rules & Guidelines

This document contains essential rules and guidelines that AI agents must follow when working on the InVisioVault project. These rules ensure consistency, maintain security standards, and preserve the project's architecture.

## 🎨 UI/UX Standards

### Theme & Appearance
- **CRITICAL**: The application is **PERMANENTLY LOCKED TO DARK MODE**
- Never add, modify, or suggest theme selection options in any UI component
- All UI components must use dark mode styling consistently
- The ConfigManager enforces theme locking - do not attempt to bypass this
- Any imported/existing configs will automatically correct theme to "dark"

### Settings Panel
- The settings panel contains these tabs: Security, Interface, Performance, Logging, Backup, Advanced
- Theme selection has been **REMOVED** from the Interface tab - never add it back
- All settings must integrate with the ConfigManager for persistence
- Validate user inputs before saving to prevent invalid configurations
- Provide clear error messages for validation failures

## 📁 Project Structure & Organization

### Documentation
- **All documentation files must be placed in the `docs/` folder**
- Never create `.md` files in the root directory (except README.md, LICENSE, etc.)
- Security-related docs belong in `docs/`
- Implementation guides belong in `docs/`
- API documentation belongs in `docs/`

### File Organization
```
InVisioVault/
├── docs/                    # All documentation files
├── core/                    # Core functionality and algorithms
├── ui/                      # All UI components and interfaces
├── utils/                   # Utilities (config, logger, etc.)
├── operations/              # File operations and processing
├── tests/                   # All test files
├── build_scripts/           # Build and deployment scripts
├── assets/                  # Static assets
└── examples/               # Example files and usage demos
```

## 🔧 Code Standards & Architecture

### Configuration Management
- **Always use ConfigManager** for any persistent settings
- Never bypass the configuration system
- The ConfigManager automatically locks theme to "dark" - respect this
- Use appropriate ConfigSection enums when accessing configs
- Validate configurations before saving
- Handle config loading failures gracefully

### Security Principles
- **Security is paramount** - never compromise security for convenience
- All steganography operations must use secure algorithms
- Implement proper error handling to prevent information leakage
- Use secure random generators for cryptographic operations
- Follow the principle of least privilege
- Validate all user inputs

### Error Handling
- Always implement comprehensive error handling
- Log errors appropriately using the Logger utility
- Provide user-friendly error messages
- Never expose sensitive information in error messages
- Handle edge cases gracefully

## 🚀 Development Practices

### Testing
- Write tests for new functionality
- Test edge cases and error conditions
- Ensure UI components work with the locked dark theme
- Test configuration changes don't bypass theme locking
- Performance test steganography operations

### Code Quality
- Follow Python PEP 8 style guidelines
- Use type hints where appropriate
- Write clear, self-documenting code
- Add docstrings to all public methods
- Keep functions focused and single-purpose

### Git & Version Control
- Write clear, descriptive commit messages
- Group related changes in single commits
- Use meaningful branch names for features
- Update documentation when making changes

## 🛡️ Security Requirements

### Steganography Implementation
- Use cryptographically secure algorithms
- Implement multiple embedding techniques (LSB, DCT, DWT, etc.)
- Add anti-detection measures (randomization, noise injection)
- Support multiple file formats securely
- Validate file integrity before and after operations

### Data Protection
- Secure deletion of temporary files
- Memory cleanup after sensitive operations
- Proper key derivation and management
- Encrypted storage of sensitive configurations

## 📋 UI Component Guidelines

### Settings Integration
- All settings must persist through ConfigManager
- Validate inputs before saving
- Provide immediate feedback for changes
- Support configuration import/export
- Include "Restore Defaults" functionality

### User Experience
- Maintain consistent dark theme across all components
- Provide clear progress indicators for long operations
- Use appropriate icons and visual cues
- Ensure accessibility standards are met
- Handle window resizing and state management

## 🔍 Specific Implementation Notes

### Banned Actions
- **NEVER** add theme selection back to the UI
- **NEVER** bypass the ConfigManager theme locking
- **NEVER** create documentation files in the root directory
- **NEVER** hardcode sensitive values or paths
- **NEVER** implement insecure steganography methods

### Required Patterns
- **ALWAYS** use the Logger utility for logging
- **ALWAYS** validate user inputs
- **ALWAYS** handle exceptions gracefully
- **ALWAYS** use ConfigManager for settings
- **ALWAYS** place documentation in `docs/` folder

## 📚 Key Files & Their Purposes

### Core Files
- `main.py` - Application entry point
- `utils/config_manager.py` - Configuration management (THEME LOCKED)
- `utils/logger.py` - Centralized logging system
- `ui/components/settings_panel.py` - Settings interface (NO THEME OPTION)

### Documentation Structure
- `docs/SECURITY_IMPROVEMENTS.md` - Security enhancement details
- `docs/STATISTICAL_MASKING_DOCUMENTATION.md` - Statistical masking guide
- `docs/api_reference.md` - API documentation
- `docs/user_guide.md` - User documentation

## 🎯 Project Goals & Vision

### Primary Objectives
- Provide secure, undetectable steganography
- Maintain user-friendly interface
- Ensure robust error handling and logging
- Support multiple file formats and algorithms
- Implement strong anti-detection measures

### Quality Standards
- High performance with large files
- Comprehensive test coverage
- Clear, maintainable code
- Extensive documentation
- Security-first approach

## 🚨 Critical Reminders

1. **THEME IS LOCKED**: Never add theme options - the app uses dark mode permanently
2. **DOCS FOLDER**: All documentation goes in `docs/` - never in root
3. **CONFIG MANAGER**: Always use it for persistent settings
4. **SECURITY FIRST**: Never compromise security for features
5. **ERROR HANDLING**: Always implement proper error handling
6. **TESTING**: Test all changes thoroughly
7. **DOCUMENTATION**: Update docs when making changes

## 📞 Decision Making

When in doubt about implementation decisions:
1. Prioritize security over convenience
2. Maintain consistency with existing patterns
3. Follow the principle of least surprise
4. Ensure changes don't break the theme lock
5. Document significant decisions
6. Test thoroughly before committing

---

**Remember**: These rules exist to maintain project quality, security, and consistency. Follow them strictly to ensure the InVisioVault project continues to meet its high standards.

**Last Updated**: 2025-01-13
**Version**: 1.0
