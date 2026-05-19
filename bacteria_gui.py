import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
import os
import sys

COLORS = {
    'bg_dark': '#1a1a1a', 
    'bg_card': '#252525',
    'bg_input': '#2d2d2d',
    'accent_pink': '#e84a7f',
    'accent_purple': '#9b59b6', 
    'accent_magenta': '#c74b8f', 
    'text_primary': '#ffffff',
    'text_secondary': '#888888',
    'text_muted': '#555555',
    'success': '#4ecca3',
    'warning': '#f39c12',
    'error': '#e74c3c',
    'border': '#3a3a3a',
    'gram_positive': '#9b59b6',
    'gram_negative': '#e84a7f', 
}

DISPLAY_NAMES = {
    'positive': 'Positive',
    'negative': 'Negative',
    'bacilli': 'Bacilli',
    'cocci_chains': 'Cocci Chains',
    'cocci_clusters': 'Cocci Clusters',
    'gp_cocci_clusters': 'Gram Positive Cocci Clusters',
    'gp_cocci_chains': 'Gram Positive Cocci Chains',
    'gp_bacilli': 'Gram Positive Bacilli',
    'gn_bacilli': 'Gram Negative Bacilli',
    'gn_cocci': 'Gram Negative Cocci',
}

def format_display_name(name):
    if name is None:
        return 'N/A'
    return DISPLAY_NAMES.get(name.lower() if isinstance(name, str) else name, name)


class ModernButton(tk.Canvas):
    def __init__(self, parent, text, command, bg=COLORS['accent_pink'], 
                 fg=COLORS['text_primary'], width=200, height=45, **kwargs):
        super().__init__(parent, width=width, height=height, 
                        bg=COLORS['bg_card'], highlightthickness=0, **kwargs)
        
        self.command = command
        self.bg_color = bg
        self.fg_color = fg
        self.text = text
        self.width = width
        self.height = height
        self.enabled = True
        
        self.draw_button()
        
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        self.bind('<Button-1>', self.on_click)
    
    def draw_button(self, hover=False):
        self.delete('all')
        
        color = self.lighten_color(self.bg_color, 20) if hover else self.bg_color
        if not self.enabled:
            color = COLORS['text_muted']
        
        r = 8
        self.create_polygon(
            r, 0, self.width-r, 0,
            self.width, 0, self.width, r,
            self.width, self.height-r, self.width, self.height,
            self.width-r, self.height, r, self.height,
            0, self.height, 0, self.height-r,
            0, r, 0, 0,
            fill=color, smooth=True
        )
        
        self.create_text(self.width/2, self.height/2, text=self.text,
                        fill=self.fg_color, font=('Segoe UI', 11, 'bold'))
    
    def lighten_color(self, color, percent):
        color = color.lstrip('#')
        r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)
        r = min(255, int(r + (255 - r) * percent / 100))
        g = min(255, int(g + (255 - g) * percent / 100))
        b = min(255, int(b + (255 - b) * percent / 100))
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def on_enter(self, event):
        if self.enabled:
            self.draw_button(hover=True)
            self.config(cursor='hand2')
    
    def on_leave(self, event):
        self.draw_button(hover=False)
        self.config(cursor='')
    
    def on_click(self, event):
        if self.enabled and self.command:
            self.command()
    
    def set_enabled(self, enabled):
        self.enabled = enabled
        self.draw_button()


class ResultCard(tk.Frame):
    
    def __init__(self, parent, title, **kwargs):
        super().__init__(parent, bg=COLORS['bg_card'], **kwargs)
        
        title_frame = tk.Frame(self, bg=COLORS['bg_card'])
        title_frame.pack(fill=tk.X, padx=15, pady=(15, 10))
        
        accent = tk.Frame(title_frame, bg=COLORS['accent_pink'], width=4, height=20)
        accent.pack(side=tk.LEFT, padx=(0, 10))
        
        self.title_label = tk.Label(title_frame, text=title,
                                    bg=COLORS['bg_card'], fg=COLORS['text_primary'],
                                    font=('Segoe UI', 12, 'bold'))
        self.title_label.pack(side=tk.LEFT)
        
        self.content_frame = tk.Frame(self, bg=COLORS['bg_card'])
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
    
    def set_accent_color(self, color):
        for widget in self.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Frame) and child.cget('width') == 4:
                        child.config(bg=color)


class CircularProgress(tk.Canvas):
    
    def __init__(self, parent, size=60, **kwargs):
        super().__init__(parent, width=size, height=size,
                        bg=COLORS['bg_dark'], highlightthickness=0, **kwargs)
        self.size = size
        self.angle = 0
        self.running = False
    
    def start(self):
        self.running = True
        self.animate()
    
    def stop(self):
        self.running = False
        self.delete('all')
    
    def animate(self):
        if not self.running:
            return
        
        self.delete('all')
        
        pad = 5
        self.create_arc(pad, pad, self.size-pad, self.size-pad,
                       start=self.angle, extent=90,
                       outline=COLORS['accent_pink'], width=3, style='arc')
        
        self.angle = (self.angle + 10) % 360
        self.after(30, self.animate)


class BacteriaGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DEEP-GRAM")
        self.root.state('zoomed')
        self.root.configure(bg=COLORS['bg_dark'])
        self.root.resizable(True, True)
        self.root.minsize(900, 700)
        
        self.pipeline = None
        self.models_loaded = False
        self.logo_image = None
        
        self.gram_model_path = r"C:\Users\carol\Desktop\DEEP-GRAM\Models\gram_vit_best_20260409_212633.pth"
        self.gp_morphology_model_path = r"C:\Users\carol\Desktop\DEEP-GRAM\Models\morphology_vit_best_20260319_193821.pth"
        self.gn_morphology_model_path = r"C:\Users\carol\Desktop\DEEP-GRAM\Models\gram - morphology_vit_best_20260408_163816.pth"
        
        self.species_model_paths = {
            'gp_cocci_clusters': r"C:\Users\carol\Desktop\DEEP-GRAM\Models\gram_pos_clusters.pth",
            'gp_cocci_chains': r"C:\Users\carol\Desktop\DEEP-GRAM\Models\gram_pos_cocci_chains_best_20260315_124115.pth",
            'gp_bacilli': r"C:\Users\carol\Desktop\DEEP-GRAM\Models\gram_pos_bacilli_best_20260316_093529.pth",
            'gn_bacilli': r"C:\Users\carol\Desktop\DEEP-GRAM\Models\gram - bacilli.pth",
        }
        
        self.logo_path = r"C:\Users\carol\Desktop\DEEP-GRAM\logo.png"
        
        self.setup_ui()
        self.root.after(100, self.load_models)
    
    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        left_col = tk.Frame(main_frame, bg=COLORS['bg_dark'], width=400)
        left_col.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left_col.pack_propagate(False)
        
        right_col = tk.Frame(main_frame, bg=COLORS['bg_dark'])
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        logo_container = tk.Frame(left_col, bg='#1a1a1a', padx=2, pady=2)
        logo_container.pack(fill=tk.X, pady=(0, 20))
        
        logo_frame = tk.Frame(logo_container, bg='#1a1a1a', height=105)
        logo_frame.pack(fill=tk.X)
        logo_frame.pack_propagate(False)
        
        self.logo_label = tk.Label(logo_frame, bg='#1a1a1a')
        self.logo_label.pack(expand=True)
        
        self.load_logo()
        
        subtitle = tk.Label(left_col, text="AI-Powered Gram Stain Analysis",
                           bg=COLORS['bg_dark'], fg=COLORS['text_secondary'],
                           font=('Segoe UI', 10))
        subtitle.pack(pady=(0, 20))
        
        status_frame = tk.Frame(left_col, bg=COLORS['bg_card'], height=50)
        status_frame.pack(fill=tk.X, pady=(0, 20))
        status_frame.pack_propagate(False)
        
        self.status_dot = tk.Canvas(status_frame, width=12, height=12,
                                   bg=COLORS['bg_card'], highlightthickness=0)
        self.status_dot.pack(side=tk.LEFT, padx=(15, 8), pady=15)
        self.status_dot.create_oval(2, 2, 10, 10, fill=COLORS['warning'], outline='')
        
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = tk.Label(status_frame, textvariable=self.status_var,
                                    bg=COLORS['bg_card'], fg=COLORS['text_secondary'],
                                    font=('Segoe UI', 10))
        self.status_label.pack(side=tk.LEFT, pady=15)
        
        self.progress = CircularProgress(status_frame, size=30)
        self.progress.pack(side=tk.RIGHT, padx=15, pady=10)
        
        mode_card = tk.Frame(left_col, bg=COLORS['bg_card'])
        mode_card.pack(fill=tk.X, pady=(0, 15))
        
        mode_header = tk.Label(mode_card, text="ANALYSIS MODE",
                              bg=COLORS['bg_card'], fg=COLORS['text_muted'],
                              font=('Segoe UI', 9, 'bold'))
        mode_header.pack(anchor=tk.W, padx=15, pady=(15, 10))
        
        self.mode_var = tk.StringVar(value="folder")
        
        modes = [
            ("Slide Folder", "folder", "Analyze multiple crops from a slide"),
            ("Single Image", "single", "Analyze a single bacteria crop")
        ]
        
        for text, value, desc in modes:
            mode_frame = tk.Frame(mode_card, bg=COLORS['bg_card'])
            mode_frame.pack(fill=tk.X, padx=15, pady=5)
            
            rb = tk.Radiobutton(mode_frame, text=text, variable=self.mode_var, value=value,
                               bg=COLORS['bg_card'], fg=COLORS['text_primary'],
                               selectcolor=COLORS['bg_input'], activebackground=COLORS['bg_card'],
                               activeforeground=COLORS['text_primary'],
                               font=('Segoe UI', 10, 'bold'),
                               highlightthickness=0)
            rb.pack(anchor=tk.W)
            
            desc_label = tk.Label(mode_frame, text=desc,
                                 bg=COLORS['bg_card'], fg=COLORS['text_muted'],
                                 font=('Segoe UI', 9))
            desc_label.pack(anchor=tk.W, padx=(20, 0))
        
        tk.Frame(mode_card, bg=COLORS['bg_card'], height=15).pack()
        
        path_card = tk.Frame(left_col, bg=COLORS['bg_card'])
        path_card.pack(fill=tk.X, pady=(0, 15))
        
        path_header = tk.Label(path_card, text="INPUT PATH",
                              bg=COLORS['bg_card'], fg=COLORS['text_muted'],
                              font=('Segoe UI', 9, 'bold'))
        path_header.pack(anchor=tk.W, padx=15, pady=(15, 10))
        
        path_input_frame = tk.Frame(path_card, bg=COLORS['bg_card'])
        path_input_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        self.path_var = tk.StringVar()
        self.path_entry = tk.Entry(path_input_frame, textvariable=self.path_var,
                                  bg=COLORS['bg_input'], fg=COLORS['text_primary'],
                                  insertbackground=COLORS['text_primary'],
                                  font=('Segoe UI', 10), relief=tk.FLAT,
                                  highlightthickness=1, highlightcolor=COLORS['accent_pink'],
                                  highlightbackground=COLORS['border'])
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10, padx=(0, 10))
        
        browse_btn = tk.Button(path_input_frame, text="Browse", command=self.browse_path,
                              bg=COLORS['bg_input'], fg=COLORS['text_primary'],
                              font=('Segoe UI', 9), relief=tk.FLAT,
                              activebackground=COLORS['accent_pink'],
                              cursor='hand2', padx=15, pady=8)
        browse_btn.pack(side=tk.RIGHT)
        
        btn_frame = tk.Frame(left_col, bg=COLORS['bg_dark'])
        btn_frame.pack(fill=tk.X, pady=(10, 20))
        
        self.analyze_btn = ModernButton(btn_frame, "ANALYZE", self.analyze,
                                       width=370, height=50)
        self.analyze_btn.pack()
        self.analyze_btn.set_enabled(False)
        
        info_frame = tk.Frame(left_col, bg=COLORS['bg_dark'])
        info_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        info_text = "DEEP-GRAM Pipeline\nGram → Morphology → Species\n13 Species | 7 Models"
        info_label = tk.Label(info_frame, text=info_text,
                             bg=COLORS['bg_dark'], fg=COLORS['text_muted'],
                             font=('Segoe UI', 9), justify=tk.CENTER)
        info_label.pack(pady=10)
        
        results_header = tk.Frame(right_col, bg=COLORS['bg_dark'])
        results_header.pack(fill=tk.X, pady=(0, 15))
        
        results_title = tk.Label(results_header, text="RESULTS",
                                bg=COLORS['bg_dark'], fg=COLORS['text_muted'],
                                font=('Segoe UI', 9, 'bold'))
        results_title.pack(side=tk.LEFT)
        
        clear_btn = tk.Button(results_header, text="Clear", command=self.clear_results,
                             bg=COLORS['bg_dark'], fg=COLORS['text_muted'],
                             font=('Segoe UI', 9), relief=tk.FLAT, bd=0,
                             activebackground=COLORS['bg_dark'],
                             activeforeground=COLORS['accent_pink'],
                             cursor='hand2')
        clear_btn.pack(side=tk.RIGHT)
        
        self.results_canvas = tk.Canvas(right_col, bg=COLORS['bg_dark'], highlightthickness=0)
        scrollbar = tk.Scrollbar(right_col, orient=tk.VERTICAL, command=self.results_canvas.yview)
        self.results_frame = tk.Frame(self.results_canvas, bg=COLORS['bg_dark'])
        
        self.results_canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas_window = self.results_canvas.create_window((0, 0), window=self.results_frame, anchor=tk.NW)
        
        self.results_frame.bind('<Configure>', lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox('all')))
        self.results_canvas.bind('<Configure>', lambda e: self.results_canvas.itemconfig(self.canvas_window, width=e.width))
        
        self.results_canvas.bind_all('<MouseWheel>', lambda e: self.results_canvas.yview_scroll(int(-1*(e.delta/120)), 'units'))
        
        self.show_welcome_message()
    
    def load_logo(self):
        try:
            if os.path.exists(self.logo_path):
                img = Image.open(self.logo_path)
                img.thumbnail((375, 100), Image.Resampling.LANCZOS)
                self.logo_image = ImageTk.PhotoImage(img)
                self.logo_label.config(image=self.logo_image, bg='#1a1a1a')
        except Exception as e:
            print(f"Could not load logo: {e}")
    
    def show_welcome_message(self):
        self.clear_results()
        
        welcome_frame = tk.Frame(self.results_frame, bg=COLORS['bg_dark'])
        welcome_frame.pack(fill=tk.BOTH, expand=True, pady=50)
        
        icon_label = tk.Label(welcome_frame, text="🔬", font=('Segoe UI', 48),
                             bg=COLORS['bg_dark'], fg=COLORS['text_muted'])
        icon_label.pack(pady=(0, 20))
        
        msg1 = tk.Label(welcome_frame, text="Ready to Analyze",
                       bg=COLORS['bg_dark'], fg=COLORS['text_primary'],
                       font=('Segoe UI', 16, 'bold'))
        msg1.pack()
        
        msg2 = tk.Label(welcome_frame, text="Select a slide folder or image to begin",
                       bg=COLORS['bg_dark'], fg=COLORS['text_muted'],
                       font=('Segoe UI', 10))
        msg2.pack(pady=(5, 0))
    
    def show_loading_message(self):
        self.clear_results()
        
        loading_frame = tk.Frame(self.results_frame, bg=COLORS['bg_dark'])
        loading_frame.pack(fill=tk.BOTH, expand=True, pady=50)
        
        spinner = CircularProgress(loading_frame, size=60)
        spinner.pack(pady=(0, 20))
        spinner.start()
        self.loading_spinner = spinner
        
        msg = tk.Label(loading_frame, text="Loading Models...",
                      bg=COLORS['bg_dark'], fg=COLORS['text_primary'],
                      font=('Segoe UI', 14))
        msg.pack()
    
    def clear_results(self):
        for widget in self.results_frame.winfo_children():
            widget.destroy()
    
    def browse_path(self):
        if self.mode_var.get() == "folder":
            path = filedialog.askdirectory(title="Select Crops Folder")
        else:
            filetypes = [('Image files', '*.jpg *.jpeg *.png *.tif *.tiff'), ('All files', '*.*')]
            path = filedialog.askopenfilename(title="Select Image", filetypes=filetypes)
        
        if path:
            self.path_var.set(path)
    
    def update_status(self, message, status='normal'):
        self.status_var.set(message)
        
        colors = {
            'success': COLORS['success'],
            'warning': COLORS['warning'],
            'error': COLORS['error'],
            'normal': COLORS['text_secondary']
        }
        
        dot_colors = {
            'success': COLORS['success'],
            'warning': COLORS['warning'],
            'error': COLORS['error'],
            'normal': COLORS['accent_pink']
        }
        
        self.status_label.config(fg=colors.get(status, COLORS['text_secondary']))
        self.status_dot.delete('all')
        self.status_dot.create_oval(2, 2, 10, 10, fill=dot_colors.get(status, COLORS['accent_pink']), outline='')
    
    def load_models(self):
        self.analyze_btn.set_enabled(False)
        self.update_status("Loading models...", 'warning')
        self.progress.start()
        self.show_loading_message()
        
        thread = threading.Thread(target=self._load_models_thread)
        thread.start()
    
    def _load_models_thread(self):
        try:
            from hierarchical_pipeline import HierarchicalPipeline, CONFIG
            
            self.pipeline = HierarchicalPipeline(CONFIG)
            
            loaded_models = []
            
            if os.path.exists(self.gram_model_path):
                self.pipeline.load_gram_model(self.gram_model_path)
                loaded_models.append("Gram")
            
            if os.path.exists(self.gp_morphology_model_path):
                self.pipeline.load_gp_morphology_model(self.gp_morphology_model_path)
                loaded_models.append("GP-Morph")
            
            if os.path.exists(self.gn_morphology_model_path):
                self.pipeline.load_gn_morphology_model(self.gn_morphology_model_path)
                loaded_models.append("GN-Morph")
            
            loaded_routes = []
            for route_name, model_path in self.species_model_paths.items():
                if os.path.exists(model_path):
                    self.pipeline.load_species_model(route_name, model_path)
                    loaded_routes.append(route_name)
            
            self.models_loaded = True
            self.root.after(0, lambda: self._models_loaded_success(loaded_models, loaded_routes))
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.root.after(0, lambda: self._models_loaded_error(error_msg))
    
    def _models_loaded_success(self, loaded_models, loaded_routes):
        self.progress.stop()
        if hasattr(self, 'loading_spinner'):
            self.loading_spinner.stop()
        
        self.analyze_btn.set_enabled(True)
        self.update_status(f"Ready | {len(loaded_models)} base + {len(loaded_routes)} species models", 'success')
        self.show_welcome_message()
    
    def _models_loaded_error(self, error):
        self.progress.stop()
        if hasattr(self, 'loading_spinner'):
            self.loading_spinner.stop()
        
        self.update_status("Failed to load models", 'error')
        self.clear_results()
        
        error_label = tk.Label(self.results_frame, text=f"Error: {error}",
                              bg=COLORS['bg_dark'], fg=COLORS['error'],
                              font=('Consolas', 9), wraplength=500, justify=tk.LEFT)
        error_label.pack(pady=20, padx=20)
    
    def analyze(self):
        path = self.path_var.get()
        if not path or not os.path.exists(path):
            self.update_status("Please select a valid path", 'error')
            return
        
        self.analyze_btn.set_enabled(False)
        self.update_status("Analyzing...", 'warning')
        self.progress.start()
        
        self.clear_results()
        analyzing_label = tk.Label(self.results_frame, text="Processing...",
                                  bg=COLORS['bg_dark'], fg=COLORS['text_secondary'],
                                  font=('Segoe UI', 12))
        analyzing_label.pack(pady=50)
        
        thread = threading.Thread(target=self._analyze_thread, args=(path,))
        thread.start()
    
    def _analyze_thread(self, path):
        try:
            if self.mode_var.get() == "folder":
                self.root.after(0, lambda: self._show_progress_ui(path))
                
                species, confidence, summary, crop_preds = self._predict_slide_with_progress(path)
                self.root.after(0, lambda: self._show_slide_results(path, species, confidence, summary))
            else:
                species, confidence, details = self.pipeline.predict_single(path)
                self.root.after(0, lambda: self._show_single_results(path, species, confidence, details))
            
            self.root.after(0, lambda: self._analysis_complete())
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.root.after(0, lambda: self._analysis_error(error_msg))
    
    def _show_progress_ui(self, path):
        self.clear_results()
        
        progress_frame = tk.Frame(self.results_frame, bg=COLORS['bg_dark'])
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=50)
        
        folder_name = os.path.basename(path)
        folder_label = tk.Label(progress_frame, text=f"Analyzing: {folder_name}",
                               bg=COLORS['bg_dark'], fg=COLORS['text_primary'],
                               font=('Segoe UI', 14, 'bold'))
        folder_label.pack(pady=(0, 20))
        
        self.progress_text_var = tk.StringVar(value="Counting crops...")
        progress_text = tk.Label(progress_frame, textvariable=self.progress_text_var,
                                bg=COLORS['bg_dark'], fg=COLORS['text_secondary'],
                                font=('Segoe UI', 11))
        progress_text.pack(pady=(0, 15))
        
        bar_container = tk.Frame(progress_frame, bg=COLORS['bg_input'], height=20, width=400)
        bar_container.pack(pady=(0, 10))
        bar_container.pack_propagate(False)
        
        self.progress_bar_fill = tk.Frame(bar_container, bg=COLORS['accent_pink'])
        self.progress_bar_fill.place(relwidth=0, relheight=1)
        
        self.progress_pct_var = tk.StringVar(value="0%")
        pct_label = tk.Label(progress_frame, textvariable=self.progress_pct_var,
                            bg=COLORS['bg_dark'], fg=COLORS['accent_pink'],
                            font=('Segoe UI', 16, 'bold'))
        pct_label.pack(pady=(0, 10))
        
        self.progress_count_var = tk.StringVar(value="0 / 0 crops")
        count_label = tk.Label(progress_frame, textvariable=self.progress_count_var,
                              bg=COLORS['bg_dark'], fg=COLORS['text_muted'],
                              font=('Segoe UI', 10))
        count_label.pack()
    
    def _update_progress(self, current, total):
        pct = current / total if total > 0 else 0
        self.progress_text_var.set(f"Classifying crops (each independently routed)...")
        self.progress_pct_var.set(f"{pct:.0%}")
        self.progress_count_var.set(f"{current:,} / {total:,} crops")
        self.progress_bar_fill.place(relwidth=pct, relheight=1)
    
    def _predict_slide_with_progress(self, slide_crops_folder, aggregation='vote'):
        from collections import Counter
        from datetime import datetime
        
        if not os.path.exists(slide_crops_folder):
            return None, 0.0, {}, []
        
        crops = [os.path.join(slide_crops_folder, f) for f in os.listdir(slide_crops_folder)
                 if f.lower().endswith('.png') or f.lower().endswith('.jpg')]
        
        if len(crops) == 0:
            return None, 0.0, {}, []
        
        total_crops = len(crops)
        self.root.after(0, lambda: self._update_progress(0, total_crops))
        
        start_time = datetime.now()
        
        all_preds = []
        gram_votes = Counter()
        morph_votes = Counter()
        route_votes = Counter()
        
        for i, crop_path in enumerate(crops):
            species, conf, details = self.pipeline.predict_single(crop_path)
            
            route = details.get('route', '') if details else ''
            if route == 'gn_cocci':
                species = 'Moraxella catarrhalis'
                conf = 1.0
            elif species is not None and 'gn_cocci' in str(species):
                species = 'Moraxella catarrhalis'
                conf = 1.0
                details['route'] = 'gn_cocci'
            
            if species is None:
                continue
            
            all_preds.append({
                'image': os.path.basename(crop_path),
                'path': crop_path,
                'species': species,
                'confidence': conf,
                **details
            })
            gram_votes[details['gram']] += 1
            morph_votes[details['morphology']] += 1
            route_votes[details['route']] += 1
            
            if (i + 1) % 10 == 0 or (i + 1) == total_crops:
                current = i + 1
                self.root.after(0, lambda c=current, t=total_crops: self._update_progress(c, t))
        
        if len(all_preds) == 0:
            return None, 0.0, {}, []
        
        if aggregation == 'vote':
            species_votes = Counter(p['species'] for p in all_preds)
            final_species = species_votes.most_common(1)[0][0]
            vote_count = species_votes.most_common(1)[0][1]
            confidence = vote_count / len(all_preds)
        else:
            species_scores = {}
            for p in all_preds:
                sp = p['species']
                if sp not in species_scores:
                    species_scores[sp] = 0
                species_scores[sp] += p['confidence']
            
            final_species = max(species_scores, key=species_scores.get)
            confidence = species_scores[final_species] / sum(species_scores.values())
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        slide_gram = gram_votes.most_common(1)[0][0] if gram_votes else 'unknown'
        slide_morphology = morph_votes.most_common(1)[0][0] if morph_votes else 'unknown'
        
        summary = {
            'total_crops': len(crops),
            'valid_crops': len(all_preds),
            'predicted_gram': slide_gram,
            'predicted_morphology': slide_morphology,
            'gram_distribution': dict(gram_votes),
            'morphology_distribution': dict(morph_votes),
            'route_distribution': dict(route_votes),
            'species_distribution': dict(Counter(p['species'] for p in all_preds)),
            'processing_time_seconds': elapsed
        }
        
        return final_species, confidence, summary, all_preds
    
    def _show_single_results(self, path, species, confidence, details):
        self.clear_results()
        
        pred_card = tk.Frame(self.results_frame, bg=COLORS['bg_card'])
        pred_card.pack(fill=tk.X, pady=(0, 15), padx=5)
        
        header = tk.Frame(pred_card, bg=COLORS['accent_pink'], height=4)
        header.pack(fill=tk.X)
        
        pred_content = tk.Frame(pred_card, bg=COLORS['bg_card'])
        pred_content.pack(fill=tk.X, padx=20, pady=20)
        
        pred_label = tk.Label(pred_content, text="PREDICTED SPECIES",
                             bg=COLORS['bg_card'], fg=COLORS['text_muted'],
                             font=('Segoe UI', 9, 'bold'))
        pred_label.pack(anchor=tk.W)
        
        species_label = tk.Label(pred_content, text=species if species else "Unknown",
                                bg=COLORS['bg_card'], fg=COLORS['text_primary'],
                                font=('Segoe UI', 24, 'bold'))
        species_label.pack(anchor=tk.W, pady=(5, 10))
        
        conf_label = tk.Label(pred_content, text=f"Confidence: {confidence:.1%}",
                             bg=COLORS['bg_card'], fg=COLORS['accent_pink'],
                             font=('Segoe UI', 12))
        conf_label.pack(anchor=tk.W)
        
        gram = details.get('gram', 'N/A')
        gram_display = format_display_name(gram)
        self._create_path_card("GRAM STAINING", 
                              gram_display.upper(),
                              f"{details.get('gram_confidence', 0):.1%}",
                              COLORS['gram_positive'] if gram == 'positive' else COLORS['gram_negative'])
        
        morph = details.get('morphology', 'N/A')
        morph_display = format_display_name(morph)
        self._create_path_card("MORPHOLOGY",
                              morph_display.upper(),
                              f"{details.get('morphology_confidence', 0):.1%}",
                              COLORS['accent_purple'])
        
        self._create_antibiotic_card(species)
        
        route = details.get('route', 'N/A')
        route_display = format_display_name(route)
        self._create_path_card("ROUTE",
                              route_display,
                              "",
                              COLORS['text_muted'])
    
    def _show_slide_results(self, path, species, confidence, summary):
        self.clear_results()
        
        if species is not None and 'gn_cocci' in str(species):
            species = 'Moraxella catarrhalis'
        
        species_dist = summary.get('species_distribution', {})
        
        cleaned_species_dist = {}
        for sp, count in species_dist.items():
            if 'gn_cocci' in str(sp):
                sp = 'Moraxella catarrhalis'
            if sp in cleaned_species_dist:
                cleaned_species_dist[sp] += count
            else:
                cleaned_species_dist[sp] = count
        species_dist = cleaned_species_dist
        species_count = species_dist.get(species, 0) if species else 0
        total_crops = summary.get('valid_crops', 0)
        
        pred_card = tk.Frame(self.results_frame, bg=COLORS['bg_card'])
        pred_card.pack(fill=tk.X, pady=(0, 15), padx=5)
        
        header = tk.Frame(pred_card, bg=COLORS['accent_pink'], height=4)
        header.pack(fill=tk.X)
        
        pred_content = tk.Frame(pred_card, bg=COLORS['bg_card'])
        pred_content.pack(fill=tk.X, padx=20, pady=20)
        
        pred_label = tk.Label(pred_content, text="PREDICTED SPECIES",
                             bg=COLORS['bg_card'], fg=COLORS['text_muted'],
                             font=('Segoe UI', 9, 'bold'))
        pred_label.pack(anchor=tk.W)
        
        species_label = tk.Label(pred_content, text=species if species else "Unknown",
                                bg=COLORS['bg_card'], fg=COLORS['text_primary'],
                                font=('Segoe UI', 24, 'bold'))
        species_label.pack(anchor=tk.W, pady=(5, 10))
        
        conf_frame = tk.Frame(pred_content, bg=COLORS['bg_card'])
        conf_frame.pack(fill=tk.X, pady=(0, 5))
        
        conf_label = tk.Label(conf_frame, text=f"Confidence: {confidence:.1%}",
                             bg=COLORS['bg_card'], fg=COLORS['accent_pink'],
                             font=('Segoe UI', 12))
        conf_label.pack(side=tk.LEFT)
        
        crops_label = tk.Label(conf_frame, text=f"{species_count:,} / {total_crops:,} crops",
                              bg=COLORS['bg_card'], fg=COLORS['text_muted'],
                              font=('Segoe UI', 10))
        crops_label.pack(side=tk.RIGHT)
        
        bar_frame = tk.Frame(pred_content, bg=COLORS['bg_input'], height=8)
        bar_frame.pack(fill=tk.X, pady=(10, 0))
        bar_frame.pack_propagate(False)
        
        bar_fill = tk.Frame(bar_frame, bg=COLORS['accent_pink'])
        bar_fill.place(relwidth=confidence, relheight=1)
        
        gram = summary.get('predicted_gram', 'N/A')
        gram_display = format_display_name(gram)
        gram_dist_formatted = {format_display_name(k): v for k, v in summary.get('gram_distribution', {}).items()}
        self._create_path_card("GRAM CONSENSUS",
                              gram_display.upper() if gram_display else 'N/A',
                              self._format_distribution(gram_dist_formatted),
                              COLORS['gram_positive'] if gram == 'positive' else COLORS['gram_negative'])
        
        morph = summary.get('predicted_morphology', 'N/A')
        morph_display = format_display_name(morph)
        morph_dist_formatted = {format_display_name(k): v for k, v in summary.get('morphology_distribution', {}).items()}
        self._create_path_card("MORPHOLOGY CONSENSUS",
                              morph_display.upper() if morph_display else 'N/A',
                              self._format_distribution(morph_dist_formatted),
                              COLORS['accent_purple'])
        
        self._create_antibiotic_card(species)
        
        self._create_distribution_card("SPECIES DISTRIBUTION",
                                       species_dist,
                                       show_counts=True,
                                       total=total_crops)
        
        route_dist = summary.get('route_distribution', {})
        route_dist_formatted = {format_display_name(k): v for k, v in route_dist.items()}
        self._create_distribution_card("ROUTE DISTRIBUTION",
                                       route_dist_formatted,
                                       COLORS['text_muted'],
                                       show_counts=True,
                                       total=total_crops)
    
    def _create_path_card(self, title, value, subtitle, accent_color):
        card = tk.Frame(self.results_frame, bg=COLORS['bg_card'])
        card.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        accent = tk.Frame(card, bg=accent_color, width=4)
        accent.pack(side=tk.LEFT, fill=tk.Y)
        
        content = tk.Frame(card, bg=COLORS['bg_card'])
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15, pady=12)
        
        title_label = tk.Label(content, text=title,
                              bg=COLORS['bg_card'], fg=COLORS['text_muted'],
                              font=('Segoe UI', 9, 'bold'))
        title_label.pack(anchor=tk.W)
        
        value_label = tk.Label(content, text=value,
                              bg=COLORS['bg_card'], fg=COLORS['text_primary'],
                              font=('Segoe UI', 14, 'bold'))
        value_label.pack(anchor=tk.W, pady=(2, 0))
        
        if subtitle:
            sub_label = tk.Label(content, text=subtitle,
                                bg=COLORS['bg_card'], fg=COLORS['text_muted'],
                                font=('Segoe UI', 9))
            sub_label.pack(anchor=tk.W)
    
    def _create_distribution_card(self, title, distribution, accent_color=None, show_counts=False, total=0):
        if not distribution:
            return
        
        card = tk.Frame(self.results_frame, bg=COLORS['bg_card'])
        card.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        header = tk.Frame(card, bg=COLORS['bg_card'])
        header.pack(fill=tk.X, padx=15, pady=(15, 10))
        
        title_label = tk.Label(header, text=title,
                              bg=COLORS['bg_card'], fg=COLORS['text_muted'],
                              font=('Segoe UI', 9, 'bold'))
        title_label.pack(anchor=tk.W)
        
        dist_total = sum(distribution.values())
        sorted_dist = sorted(distribution.items(), key=lambda x: -x[1])
        
        for name, count in sorted_dist:
            pct = count / dist_total if dist_total > 0 else 0
            
            row = tk.Frame(card, bg=COLORS['bg_card'])
            row.pack(fill=tk.X, padx=15, pady=3)
            
            display_name = name[:25] + "..." if len(str(name)) > 28 else name
            name_label = tk.Label(row, text=display_name, width=28, anchor=tk.W,
                                 bg=COLORS['bg_card'], fg=COLORS['text_primary'],
                                 font=('Segoe UI', 9))
            name_label.pack(side=tk.LEFT)
            
            bar_container = tk.Frame(row, bg=COLORS['bg_input'], height=16)
            bar_container.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
            bar_container.pack_propagate(False)
            
            color = accent_color if accent_color else COLORS['accent_pink']
            bar = tk.Frame(bar_container, bg=color)
            bar.place(relwidth=pct, relheight=1)
            
            if show_counts and total > 0:
                count_text = f"{count:,} / {total:,}"
                count_label = tk.Label(row, text=count_text, width=12, anchor=tk.E,
                                      bg=COLORS['bg_card'], fg=COLORS['text_muted'],
                                      font=('Segoe UI', 9))
                count_label.pack(side=tk.RIGHT)
            else:
                pct_label = tk.Label(row, text=f"{pct:.1%}", width=6, anchor=tk.E,
                                    bg=COLORS['bg_card'], fg=COLORS['text_muted'],
                                    font=('Segoe UI', 9))
                pct_label.pack(side=tk.RIGHT)
        
        tk.Frame(card, bg=COLORS['bg_card'], height=10).pack()
    
    def _format_distribution(self, dist):
        if not dist:
            return ""
        total = sum(dist.values())
        parts = [f"{k}: {100*v/total:.0f}%" for k, v in sorted(dist.items(), key=lambda x: -x[1])]
        return " | ".join(parts)
    
    def _analysis_complete(self):
        self.progress.stop()
        self.analyze_btn.set_enabled(True)
        self.update_status("Analysis complete", 'success')
    
    def _analysis_error(self, error):
        self.progress.stop()
        self.analyze_btn.set_enabled(True)
        self.update_status("Analysis failed", 'error')
        
        self.clear_results()
        error_label = tk.Label(self.results_frame, text=f"Error:\n\n{error}",
                              bg=COLORS['bg_dark'], fg=COLORS['error'],
                              font=('Consolas', 9), wraplength=500, justify=tk.LEFT)
        error_label.pack(pady=20, padx=20)
    
    def _get_antibiotic_recommendations(self, species):
        ANTIBIOTIC_RECOMMENDATIONS = {
            'Staphylococcus aureus': {
                'first_line': ['Nafcillin', 'Oxacillin', 'Cefazolin'],
                'alternatives': ['Vancomycin', 'Daptomycin', 'Linezolid', 'Clindamycin'],
            },
            'Staphylococcus epidermidis': {
                'first_line': ['Vancomycin', 'Daptomycin'],
                'alternatives': ['Linezolid', 'Rifampin (combination)', 'TMP-SMX'],
            },
            'Enterococcus faecalis': {
                'first_line': ['Ampicillin', 'Penicillin G'],
                'alternatives': ['Vancomycin', 'Linezolid', 'Daptomycin'],
            },
            'Enterococcus faecium': {
                'first_line': ['Linezolid', 'Daptomycin'],
                'alternatives': ['Quinupristin-dalfopristin', 'Tigecycline'],
            },
            'Streptococcus pyogenes': {
                'first_line': ['Penicillin G', 'Amoxicillin'],
                'alternatives': ['Cephalexin', 'Azithromycin', 'Clindamycin'],
            },
            'Streptococcus pneumoniae': {
                'first_line': ['Amoxicillin', 'Penicillin G'],
                'alternatives': ['Ceftriaxone', 'Levofloxacin', 'Vancomycin'],
            },
            'Streptococcus mitis': {
                'first_line': ['Penicillin G', 'Ampicillin'],
                'alternatives': ['Ceftriaxone', 'Vancomycin'],
            },
            'Corynebacterium striatum': {
                'first_line': ['Vancomycin'],
                'alternatives': ['Linezolid', 'Daptomycin'],
            },
            'Listeria monocytogenes': {
                'first_line': ['Ampicillin', 'Penicillin G'],
                'alternatives': ['TMP-SMX', 'Meropenem'],
            },
            'Klebsiella pneumoniae': {
                'first_line': ['Ceftriaxone', 'Cefepime'],
                'alternatives': ['Meropenem', 'Piperacillin-tazobactam', 'Ciprofloxacin'],
            },
            'Enterobacter cloacae': {
                'first_line': ['Cefepime', 'Meropenem'],
                'alternatives': ['Ciprofloxacin', 'Piperacillin-tazobactam', 'TMP-SMX'],
            },
            'Pseudomonas aeruginosa': {
                'first_line': ['Piperacillin-tazobactam', 'Cefepime', 'Ceftazidime'],
                'alternatives': ['Meropenem', 'Ciprofloxacin', 'Aztreonam', 'Tobramycin'],
            },
            'Moraxella catarrhalis': {
                'first_line': ['Amoxicillin-clavulanate', 'Azithromycin'],
                'alternatives': ['Cefuroxime', 'TMP-SMX', 'Fluoroquinolones'],
            },
        }
        
        if species and species in ANTIBIOTIC_RECOMMENDATIONS:
            return ANTIBIOTIC_RECOMMENDATIONS[species]
        return None
    
    def _create_antibiotic_card(self, species):
        recommendations = self._get_antibiotic_recommendations(species)
        
        if not recommendations:
            return
        
        card = tk.Frame(self.results_frame, bg=COLORS['bg_card'])
        card.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        header_frame = tk.Frame(card, bg=COLORS['success'], height=4)
        header_frame.pack(fill=tk.X)
        
        title_frame = tk.Frame(card, bg=COLORS['bg_card'])
        title_frame.pack(fill=tk.X, padx=15, pady=(15, 10))
        
        title_label = tk.Label(title_frame, text="ANTIBIOTIC RECOMMENDATIONS",
                              bg=COLORS['bg_card'], fg=COLORS['success'],
                              font=('Segoe UI', 9, 'bold'))
        title_label.pack(anchor=tk.W)
        
        first_frame = tk.Frame(card, bg=COLORS['bg_card'])
        first_frame.pack(fill=tk.X, padx=15, pady=(0, 5))
        
        first_header = tk.Label(first_frame, text="First-Line:",
                               bg=COLORS['bg_card'], fg=COLORS['accent_pink'],
                               font=('Segoe UI', 10, 'bold'))
        first_header.pack(anchor=tk.W)
        
        first_drugs = tk.Label(first_frame, text="  •  " + "\n  •  ".join(recommendations['first_line']),
                              bg=COLORS['bg_card'], fg=COLORS['text_primary'],
                              font=('Segoe UI', 10), justify=tk.LEFT)
        first_drugs.pack(anchor=tk.W, padx=(10, 0))
        
        alt_frame = tk.Frame(card, bg=COLORS['bg_card'])
        alt_frame.pack(fill=tk.X, padx=15, pady=(5, 15))
        
        alt_header = tk.Label(alt_frame, text="Alternatives:",
                             bg=COLORS['bg_card'], fg=COLORS['accent_purple'],
                             font=('Segoe UI', 10, 'bold'))
        alt_header.pack(anchor=tk.W)
        
        alt_drugs = tk.Label(alt_frame, text="  •  " + "\n  •  ".join(recommendations['alternatives']),
                            bg=COLORS['bg_card'], fg=COLORS['text_primary'],
                            font=('Segoe UI', 10), justify=tk.LEFT)
        alt_drugs.pack(anchor=tk.W, padx=(10, 0))
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = BacteriaGUI()
    app.run()
