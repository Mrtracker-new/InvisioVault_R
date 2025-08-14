#!/usr/bin/env python3
"""
InVisioVault Self-Executing Image Viewer
Custom viewer for images with embedded executable content.

Author: Rolan (RNR)
Purpose: Educational demonstration of self-executing images
"""

import sys
import os
from pathlib import Path
from tkinter import messagebox, filedialog
import tkinter as tk
from tkinter import ttk

# Add InVisioVault to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from core.self_executing_engine import SelfExecutingEngine
    from utils.logger import Logger
except ImportError as e:
    print(f"Error importing InVisioVault modules: {e}")
    print("Please ensure you're running this from the InVisioVault directory.")
    sys.exit(1)


class SelfExecutingViewer:
    """Custom viewer for self-executing images."""
    
    def __init__(self):
        self.engine = SelfExecutingEngine()
        self.logger = Logger()
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("InVisioVault Self-Executing Image Viewer")
        self.root.geometry("600x500")
        self.root.minsize(500, 400)
        
        # Set window icon (if available)
        try:
            icon_path = Path(__file__).parent / "assets" / "icons" / "app.ico"
            if icon_path.exists():
                self.root.iconbitmap(str(icon_path))
        except Exception:
            pass
        
        self.setup_ui()
        self.logger.info("Self-executing image viewer initialized")
    
    def setup_ui(self):
        """Setup the viewer UI."""
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(2, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        
        title_label = ttk.Label(
            header_frame, 
            text="🚀 Self-Executing Image Viewer",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0)
        
        subtitle_label = ttk.Label(
            header_frame,
            text="Analyze and execute embedded code from steganographic images",
            font=("Arial", 9)
        )
        subtitle_label.grid(row=1, column=0, pady=(5, 0))
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="Image Selection", padding="10")
        file_frame.grid(row=1, column=0, sticky="ew", pady=(0, 20))
        file_frame.grid_columnconfigure(0, weight=1)
        
        # File path entry
        self.file_path_var = tk.StringVar()
        file_entry_frame = ttk.Frame(file_frame)
        file_entry_frame.grid(row=0, column=0, sticky="ew")
        file_entry_frame.grid_columnconfigure(0, weight=1)
        
        self.file_entry = ttk.Entry(
            file_entry_frame, 
            textvariable=self.file_path_var,
            font=("Consolas", 9)
        )
        self.file_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        
        browse_btn = ttk.Button(
            file_entry_frame,
            text="Browse...",
            command=self.browse_image
        )
        browse_btn.grid(row=0, column=1)
        
        # Password entry
        password_frame = ttk.Frame(file_frame)
        password_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        password_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(password_frame, text="Password:").grid(row=0, column=0, padx=(0, 10))
        
        self.password_var = tk.StringVar()
        self.password_entry = ttk.Entry(
            password_frame,
            textvariable=self.password_var,
            show="*",
            font=("Consolas", 9)
        )
        self.password_entry.grid(row=0, column=1, sticky="ew")
        
        # Action buttons
        action_frame = ttk.Frame(file_frame)
        action_frame.grid(row=2, column=0, pady=(15, 0))
        
        analyze_btn = ttk.Button(
            action_frame,
            text="🔍 Analyze Image",
            command=self.analyze_image
        )
        analyze_btn.grid(row=0, column=0, padx=(0, 10))
        
        execute_btn = ttk.Button(
            action_frame,
            text="▶️ Execute Content",
            command=self.execute_content
        )
        execute_btn.grid(row=0, column=1)
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=2, column=0, sticky="nsew")
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)
        
        # Results text area with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.grid(row=0, column=0, sticky="nsew")
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
        
        self.results_text = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            state=tk.DISABLED
        )
        self.results_text.grid(row=0, column=0, sticky="nsew")
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=3, column=0, sticky="ew", pady=(20, 0))
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.grid(row=0, column=0)
        
        # Menu bar
        self.setup_menu()
        
        # Bind events
        self.root.bind('<Control-o>', lambda e: self.browse_image())
        self.root.bind('<Control-a>', lambda e: self.analyze_image())
        self.root.bind('<F5>', lambda e: self.execute_content())
        
        # Initial message
        self.update_results("Welcome to the Self-Executing Image Viewer!\n\n"
                           "Select an image file and click 'Analyze Image' to check for embedded executable content.\n\n"
                           "⚠️ WARNING: Only execute content from trusted sources!\n\n"
                           "Keyboard shortcuts:\n"
                           "  Ctrl+O - Browse for image\n"
                           "  Ctrl+A - Analyze image\n"
                           "  F5 - Execute content")
    
    def setup_menu(self):
        """Setup menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image...", accelerator="Ctrl+O", command=self.browse_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Analyze Image", accelerator="Ctrl+A", command=self.analyze_image)
        tools_menu.add_command(label="Execute Content", accelerator="F5", command=self.execute_content)
        tools_menu.add_separator()
        tools_menu.add_command(label="Clear Results", command=self.clear_results)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def browse_image(self):
        """Browse for an image file."""
        file_path = filedialog.askopenfilename(
            title="Select Self-Executing Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("BMP files", "*.bmp"),
                ("TIFF files", "*.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.file_path_var.set(file_path)
            self.update_status(f"Selected: {Path(file_path).name}")
            self.clear_results()
    
    def analyze_image(self):
        """Analyze the selected image for executable content."""
        image_path = self.file_path_var.get().strip()
        password = self.password_var.get().strip() or None
        
        if not image_path:
            messagebox.showwarning("No Image Selected", "Please select an image file to analyze.")
            return
        
        if not Path(image_path).exists():
            messagebox.showerror("File Not Found", f"The selected image file does not exist:\n{image_path}")
            return
        
        try:
            self.update_status("Analyzing image...")
            self.clear_results()
            
            # Analyze in safe mode
            result = self.engine.extract_and_execute(
                image_path=image_path,
                password=password,
                execution_mode='safe'
            )
            
            # Format and display results
            if result.get('success'):
                self.update_results(f"""✅ EXECUTABLE CONTENT DETECTED!

🔍 Analysis Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 Image File: {Path(image_path).name}
📊 Content Type: {result.get('type', 'Unknown').upper()}
🎯 Script Type: {result.get('script_type', 'N/A')}
⚡ Can Execute: {'Yes' if result.get('can_execute') else 'No'}
🔄 Auto-Execute: {'Yes' if result.get('auto_execute') else 'No'}

📝 Details:
{result.get('message', 'No additional details available')}

⚠️ SECURITY WARNING:
This image contains executable content that could potentially
be dangerous if executed. Only run content from trusted sources!

💡 Next Steps:
• Review the embedded content carefully
• Use 'Execute Content' button to run (with caution)
• Check the password if extraction failed
""")
                self.update_status("✅ Executable content detected!")
                
            else:
                error_msg = result.get('error', result.get('message', 'Unknown error'))
                self.update_results(f"""❌ NO EXECUTABLE CONTENT FOUND

🔍 Analysis Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 Image File: {Path(image_path).name}
📊 Status: No embedded executable content detected

📝 Details:
{error_msg}

💡 Possible reasons:
• The image doesn't contain hidden executable content
• Wrong password (if content is encrypted)
• Unsupported format or encoding method
• Content was created with different tools

🎯 Try:
• Different password if you suspect encrypted content
• Verify the image was created with InVisioVault tools
• Check if it's a polyglot file (try running it directly)
""")
                self.update_status("❌ No executable content found")
                
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.logger.error(error_msg)
            self.update_results(f"""❌ ANALYSIS ERROR

🔍 Error Details:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 Image File: {Path(image_path).name}
❌ Error: {str(e)}

💡 Troubleshooting:
• Ensure the image file is not corrupted
• Check file permissions
• Verify the image format is supported (PNG, BMP, TIFF)
• Try a different image or contact support
""")
            self.update_status(f"❌ Analysis error: {str(e)}")
    
    def execute_content(self):
        """Execute the embedded content."""
        image_path = self.file_path_var.get().strip()
        password = self.password_var.get().strip() or None
        
        if not image_path:
            messagebox.showwarning("No Image Selected", "Please select an image file first.")
            return
        
        if not Path(image_path).exists():
            messagebox.showerror("File Not Found", f"The selected image file does not exist:\n{image_path}")
            return
        
        # Confirm execution
        response = messagebox.askyesno(
            "⚠️ Confirm Execution",
            "Are you sure you want to execute the embedded content?\n\n"
            "WARNING: This could potentially be dangerous!\n"
            "Only proceed if you trust the source of this image.\n\n"
            "Do you want to continue?",
            icon='warning'
        )
        
        if not response:
            self.update_status("Execution cancelled by user")
            return
        
        try:
            self.update_status("Executing embedded content...")
            
            # Execute in auto mode
            result = self.engine.extract_and_execute(
                image_path=image_path,
                password=password,
                execution_mode='auto'
            )
            
            # Display execution results
            if result.get('success'):
                stdout = result.get('stdout', 'No output produced')
                stderr = result.get('stderr', 'None')
                return_code = result.get('return_code', 'N/A')
                
                self.update_results(f"""✅ EXECUTION COMPLETED SUCCESSFULLY!

🚀 Execution Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 Image File: {Path(image_path).name}
🎯 Content Type: {result.get('type', 'Unknown').upper()}
📊 Return Code: {return_code}

📤 Program Output:
{stdout}

⚠️ Error Output:
{stderr}

💡 Execution completed successfully!
""")
                self.update_status("✅ Execution completed successfully")
                
            else:
                error_msg = result.get('error', 'Unknown execution error')
                self.update_results(f"""❌ EXECUTION FAILED!

🚀 Execution Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 Image File: {Path(image_path).name}
❌ Error: {error_msg}

💡 Possible causes:
• No executable content found in the image
• Wrong password for encrypted content  
• Runtime error in the embedded script
• Missing interpreter (Python, Node.js, etc.)
• Execution timeout or permission issues

🎯 Troubleshooting:
• Analyze the image first to verify content exists
• Check that required interpreters are installed
• Try with correct password if content is encrypted
""")
                self.update_status(f"❌ Execution failed: {error_msg}")
                
        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            self.logger.error(error_msg)
            self.update_results(f"""❌ EXECUTION ERROR!

🚀 Error Details:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 Image File: {Path(image_path).name}
❌ Error: {str(e)}

💡 This could be due to:
• System security restrictions
• Missing dependencies or interpreters  
• Corrupted embedded content
• Insufficient permissions
""")
            self.update_status(f"❌ Execution error: {str(e)}")
    
    def clear_results(self):
        """Clear the results text area."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
    
    def update_results(self, text):
        """Update the results text area."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, text)
        self.results_text.config(state=tk.DISABLED)
        self.results_text.see(1.0)  # Scroll to top
    
    def update_status(self, text):
        """Update the status bar."""
        self.status_var.set(text)
        self.root.update_idletasks()
    
    def show_about(self):
        """Show about dialog."""
        about_text = """InVisioVault Self-Executing Image Viewer
Version 1.0.0

A specialized viewer for analyzing and executing
embedded code from steganographic images.

Author: Rolan (RNR)
Purpose: Educational steganography research

⚠️ WARNING:
This tool is for educational purposes only.
Always exercise extreme caution when executing
embedded content from unknown sources!

Features:
• Safe analysis mode
• Multiple script type support
• Encrypted content handling
• Detailed execution logging

© 2025 Rolan (RNR) - Educational Project"""
        
        messagebox.showinfo("About Self-Executing Image Viewer", about_text)
    
    def run(self):
        """Run the viewer application."""
        try:
            # Center window on screen
            self.root.update_idletasks()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            x = (self.root.winfo_screenwidth() // 2) - (width // 2)
            y = (self.root.winfo_screenheight() // 2) - (height // 2)
            self.root.geometry(f'{width}x{height}+{x}+{y}')
            
            self.root.mainloop()
            
        except KeyboardInterrupt:
            print("\nViewer terminated by user.")
        except Exception as e:
            self.logger.error(f"Viewer error: {e}")
            messagebox.showerror("Viewer Error", f"An error occurred:\n{e}")


def main():
    """Main function for standalone execution."""
    print("🚀 InVisioVault Self-Executing Image Viewer")
    print("=" * 50)
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        print(f"📁 Opening image: {sys.argv[1]}")
        
        # Command line mode - analyze directly
        engine = SelfExecutingEngine()
        result = engine.extract_and_execute(sys.argv[1], execution_mode='interactive')
        
        if result.get('success'):
            print("✅ Analysis completed!")
            print(f"🎯 Type: {result.get('type')}")
            print(f"📝 Details: {result.get('message')}")
        else:
            print("❌ No executable content found or analysis failed.")
            print(f"📝 Details: {result.get('message', result.get('error', 'Unknown'))}")
        
        return
    
    # GUI mode
    try:
        print("🖥️  Starting GUI mode...")
        viewer = SelfExecutingViewer()
        viewer.run()
        
    except ImportError as e:
        print(f"❌ GUI dependencies not available: {e}")
        print("💡 Try installing tkinter or use command line mode:")
        print("   python self_executing_viewer.py <image_file>")
        
    except Exception as e:
        print(f"❌ Failed to start viewer: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
