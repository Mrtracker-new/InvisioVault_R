#!/usr/bin/env python3
# InVisioVault Self-Executing Image Viewer
# Custom viewer for images with embedded executable content.

import sys
import os
from pathlib import Path
from tkinter import messagebox, filedialog
import tkinter as tk

# Add InVisioVault to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from core.self_executing_engine import SelfExecutingEngine
    from utils.logger import Logger
except ImportError as e:
    print(f'Error importing InVisioVault modules: {e}')
    print('Please ensure you are running this from the InVisioVault directory.')
    sys.exit(1)

class SelfExecutingViewer:
    def __init__(self):
        self.engine = SelfExecutingEngine()
        self.root = tk.Tk()
        self.root.title('InVisioVault Self-Executing Image Viewer')
        self.setup_ui()
    
    def setup_ui(self):
        frame = tk.Frame(self.root, padx=20, pady=20)
        frame.pack()
        
        tk.Label(frame, text='Self-Executing Image Viewer', 
                font=('Arial', 16, 'bold')).pack(pady=10)
        
        tk.Button(frame, text='Open Image', command=self.open_image,
                 width=20, height=2).pack(pady=5)
        
        tk.Button(frame, text='Exit', command=self.root.quit,
                 width=20, height=2).pack(pady=5)
    
    def open_image(self):
        file_path = filedialog.askopenfilename(
            title='Select Self-Executing Image',
            filetypes=[
                ('Image files', '*.png *.jpg *.jpeg *.bmp *.tiff'),
                ('All files', '*.*')
            ]
        )
        
        if file_path:
            self.analyze_image(file_path)
    
    def analyze_image(self, file_path):
        result = self.engine.extract_and_execute(file_path, execution_mode='safe')
        
        if result.get('success'):
            message = f'Executable content detected!\n\nType: {result.get("type")}\nDetails: {result.get("message")}'
            
            if messagebox.askyesno('Execute?', message + '\n\nExecute the embedded content?'):
                exec_result = self.engine.extract_and_execute(file_path, execution_mode='auto')
                self.show_execution_result(exec_result)
        else:
            messagebox.showinfo('Analysis Result', 
                              result.get('message', 'No executable content found'))
    
    def show_execution_result(self, result):
        if result.get('success'):
            message = f'Execution completed successfully!\n\nOutput: {result.get("stdout", "No output")}'
        else:
            message = f'Execution failed!\n\nError: {result.get("error", "Unknown error")}'
        
        messagebox.showinfo('Execution Result', message)
    
    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Command line mode
        image_path = sys.argv[1]
        engine = SelfExecutingEngine()
        result = engine.extract_and_execute(image_path, execution_mode='interactive')
        print(f'Result: {result}')
    else:
        # GUI mode
        viewer = SelfExecutingViewer()
        viewer.run()