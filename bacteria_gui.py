"""
DEEP-GRAM - Bacterial Species Identification from Gram Stains
==============================================================
Professional GUI for hierarchical classification pipeline.

Routing Structure:
  Gram Model → 
    ├── Positive → GP Morphology Model →
    │     ├── cocci_clusters → GP Cocci Clusters Species (SA, SE)
    │     ├── cocci_chains → GP Cocci Chains Species (SPY, SPN, VS, EFS, EF)
    │     └── bacilli → GP Bacilli Species (CB, LS)
    │
    └── Negative → GN Morphology Model →
          ├── bacilli → GN Bacilli Species (KP, ECL, PA, AB)
          └── cocci_clusters → GN Cocci Species (MC, HI)

Usage:
    python bacteria_gui.py
"""

import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import threading
import os
import sys

# Color scheme
COLORS = {
    'bg_dark': '#1a1a2e',
    'bg_medium': '#16213e',
    'bg_light': '#0f3460',
    'accent': '#e94560',
    'accent_hover': '#ff6b6b',
    'text_primary': '#ffffff',
    'text_secondary': '#a0a0a0',
    'success': '#4ecca3',
    'warning': '#ffc857',
    'error': '#e94560',
    'card_bg': '#1f2940',
}


class BacteriaGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DEEP-GRAM")
        self.root.geometry("900x750")
        self.root.configure(bg=COLORS['bg_dark'])
        self.root.resizable(True, True)
        
        # Set minimum window size
        self.root.minsize(800, 700)
        
        self.pipeline = None
        self.models_loaded = False
        
        # ====================================================================
        # UPDATE THESE PATHS TO YOUR TRAINED MODELS
        # ====================================================================
        self.model_dir = r"C:\Gram Stain Training Data\models"
        
        # Level 1: Gram model (ViT)
        self.gram_model_path = r"C:\Gram Stain Training Data\models\Gram\gram_vit_best_20260316_175955.pth"
        
        # Level 2: Morphology models (ViT) - separate for GP and GN
        self.gp_morphology_model_path = r"C:\Gram Stain Training Data\models\gram pos\morphology_vit_best_20260319_193821.pth"
        self.gn_morphology_model_path = r"C:\Gram Stain Training Data\models\gram neg\morphology_vit_best_20260320_014219.pth"
        
        # Level 3: Species models (one per route)
        self.species_model_paths = {
            'gp_cocci_clusters': r"C:\Gram Stain Training Data\models\Gram Pos Cocci Clusters\gram_pos_clusters.pth",
            'gp_cocci_chains': r"C:\Gram Stain Training Data\models\Gram Pos Cocci Chains\gram_pos_cocci_chains_best_20260315_124115.pth",
            'gp_bacilli': r"C:\Gram Stain Training Data\models\Gram Pos Bacilli\gram_pos_bacilli_best_20260316_093529.pth",
            'gn_bacilli': r"C:\Gram Stain Training Data\models\Gram Neg Bacilli\gram_neg_bacilli_best_20260315_083614.pth",
            'gn_cocci': r"C:\Gram Stain Training Data\models\Gram Neg Cocci\gram_neg_cocci.pth",
        }
        
        self.setup_styles()
        self.setup_ui()
        
        # Auto-load models on startup
        self.root.after(100, self.load_models)
    
    def setup_styles(self):
        """Configure ttk styles for modern look."""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure frame styles
        self.style.configure('Dark.TFrame', background=COLORS['bg_dark'])
        self.style.configure('Card.TFrame', background=COLORS['card_bg'])
        
        # Configure label styles
        self.style.configure('Title.TLabel',
                           background=COLORS['bg_dark'],
                           foreground=COLORS['text_primary'],
                           font=('Segoe UI', 28, 'bold'))
        
        self.style.configure('Subtitle.TLabel',
                           background=COLORS['bg_dark'],
                           foreground=COLORS['text_secondary'],
                           font=('Segoe UI', 11))
        
        self.style.configure('Header.TLabel',
                           background=COLORS['card_bg'],
                           foreground=COLORS['text_primary'],
                           font=('Segoe UI', 12, 'bold'))
        
        self.style.configure('Status.TLabel',
                           background=COLORS['bg_dark'],
                           foreground=COLORS['text_secondary'],
                           font=('Segoe UI', 10))
        
        self.style.configure('Success.TLabel',
                           background=COLORS['bg_dark'],
                           foreground=COLORS['success'],
                           font=('Segoe UI', 10, 'bold'))
        
        self.style.configure('Error.TLabel',
                           background=COLORS['bg_dark'],
                           foreground=COLORS['error'],
                           font=('Segoe UI', 10, 'bold'))
        
        self.style.configure('Warning.TLabel',
                           background=COLORS['bg_dark'],
                           foreground=COLORS['warning'],
                           font=('Segoe UI', 10, 'bold'))
        
        # Configure button styles
        self.style.configure('Accent.TButton',
                           background=COLORS['accent'],
                           foreground=COLORS['text_primary'],
                           font=('Segoe UI', 11, 'bold'),
                           padding=(20, 10))
        
        self.style.map('Accent.TButton',
                      background=[('active', COLORS['accent_hover']),
                                ('disabled', COLORS['bg_light'])])
        
        self.style.configure('Secondary.TButton',
                           background=COLORS['bg_light'],
                           foreground=COLORS['text_primary'],
                           font=('Segoe UI', 10),
                           padding=(15, 8))
        
        self.style.map('Secondary.TButton',
                      background=[('active', COLORS['bg_medium'])])
        
        # Configure radiobutton styles
        self.style.configure('Dark.TRadiobutton',
                           background=COLORS['card_bg'],
                           foreground=COLORS['text_primary'],
                           font=('Segoe UI', 10))
        
        # Configure entry styles
        self.style.configure('Dark.TEntry',
                           fieldbackground=COLORS['bg_medium'],
                           foreground=COLORS['text_primary'],
                           insertcolor=COLORS['text_primary'])
        
        # Configure progressbar
        self.style.configure('Accent.Horizontal.TProgressbar',
                           background=COLORS['accent'],
                           troughcolor=COLORS['bg_medium'])
    
    def setup_ui(self):
        """Build the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, style='Dark.TFrame', padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ==================== HEADER ====================
        header_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Title with icon
        title_frame = ttk.Frame(header_frame, style='Dark.TFrame')
        title_frame.pack()
        
        title_label = ttk.Label(title_frame, text="DEEP-GRAM", style='Title.TLabel')
        title_label.pack()
        
        subtitle_label = ttk.Label(header_frame, 
                                   text="Deep Learning for Gram Stain Analysis & Bacterial Species Identification",
                                   style='Subtitle.TLabel')
        subtitle_label.pack(pady=(5, 0))
        
        # ==================== STATUS BAR ====================
        status_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, style='Warning.TLabel')
        self.status_label.pack()
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', 
                                        style='Accent.Horizontal.TProgressbar',
                                        length=400)
        self.progress.pack(pady=(10, 0))
        
        # ==================== CONTROL PANEL ====================
        control_card = tk.Frame(main_frame, bg=COLORS['card_bg'], padx=20, pady=15)
        control_card.pack(fill=tk.X, pady=(0, 15))
        
        # Analysis Mode
        mode_frame = tk.Frame(control_card, bg=COLORS['card_bg'])
        mode_frame.pack(fill=tk.X, pady=(0, 15))
        
        mode_label = tk.Label(mode_frame, text="Analysis Mode", 
                             bg=COLORS['card_bg'], fg=COLORS['text_primary'],
                             font=('Segoe UI', 11, 'bold'))
        mode_label.pack(anchor=tk.W)
        
        self.mode_var = tk.StringVar(value="folder")
        
        radio_frame = tk.Frame(mode_frame, bg=COLORS['card_bg'])
        radio_frame.pack(anchor=tk.W, pady=(8, 0))
        
        folder_radio = tk.Radiobutton(radio_frame, text="Slide Folder (multiple crops)", 
                                      variable=self.mode_var, value="folder",
                                      bg=COLORS['card_bg'], fg=COLORS['text_primary'],
                                      selectcolor=COLORS['bg_medium'],
                                      activebackground=COLORS['card_bg'],
                                      activeforeground=COLORS['text_primary'],
                                      font=('Segoe UI', 10))
        folder_radio.pack(side=tk.LEFT, padx=(0, 20))
        
        single_radio = tk.Radiobutton(radio_frame, text="Single Image", 
                                      variable=self.mode_var, value="single",
                                      bg=COLORS['card_bg'], fg=COLORS['text_primary'],
                                      selectcolor=COLORS['bg_medium'],
                                      activebackground=COLORS['card_bg'],
                                      activeforeground=COLORS['text_primary'],
                                      font=('Segoe UI', 10))
        single_radio.pack(side=tk.LEFT)
        
        # Path Selection
        path_frame = tk.Frame(control_card, bg=COLORS['card_bg'])
        path_frame.pack(fill=tk.X, pady=(0, 15))
        
        path_label = tk.Label(path_frame, text="Input Path", 
                             bg=COLORS['card_bg'], fg=COLORS['text_primary'],
                             font=('Segoe UI', 11, 'bold'))
        path_label.pack(anchor=tk.W)
        
        path_input_frame = tk.Frame(path_frame, bg=COLORS['card_bg'])
        path_input_frame.pack(fill=tk.X, pady=(8, 0))
        
        self.path_var = tk.StringVar()
        self.path_entry = tk.Entry(path_input_frame, textvariable=self.path_var,
                                   bg=COLORS['bg_medium'], fg=COLORS['text_primary'],
                                   insertbackground=COLORS['text_primary'],
                                   font=('Segoe UI', 10), relief=tk.FLAT,
                                   highlightthickness=1, highlightcolor=COLORS['accent'],
                                   highlightbackground=COLORS['bg_light'])
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))
        
        browse_btn = tk.Button(path_input_frame, text="Browse", command=self.browse_path,
                              bg=COLORS['bg_light'], fg=COLORS['text_primary'],
                              font=('Segoe UI', 10), relief=tk.FLAT,
                              activebackground=COLORS['accent'],
                              activeforeground=COLORS['text_primary'],
                              cursor='hand2', padx=15, pady=5)
        browse_btn.pack(side=tk.RIGHT)
        
        # Analyze Button
        button_frame = tk.Frame(control_card, bg=COLORS['card_bg'])
        button_frame.pack(fill=tk.X)
        
        self.analyze_btn = tk.Button(button_frame, text="ANALYZE", command=self.analyze,
                                    bg=COLORS['accent'], fg=COLORS['text_primary'],
                                    font=('Segoe UI', 12, 'bold'), relief=tk.FLAT,
                                    activebackground=COLORS['accent_hover'],
                                    activeforeground=COLORS['text_primary'],
                                    cursor='hand2', padx=30, pady=10,
                                    state='disabled')
        self.analyze_btn.pack(pady=(5, 0))
        
        # ==================== RESULTS PANEL ====================
        results_card = tk.Frame(main_frame, bg=COLORS['card_bg'], padx=20, pady=15)
        results_card.pack(fill=tk.BOTH, expand=True)
        
        results_header = tk.Frame(results_card, bg=COLORS['card_bg'])
        results_header.pack(fill=tk.X, pady=(0, 10))
        
        results_label = tk.Label(results_header, text="Results", 
                                bg=COLORS['card_bg'], fg=COLORS['text_primary'],
                                font=('Segoe UI', 11, 'bold'))
        results_label.pack(side=tk.LEFT)
        
        # Clear button
        clear_btn = tk.Button(results_header, text="Clear", command=self.clear_results,
                             bg=COLORS['bg_light'], fg=COLORS['text_secondary'],
                             font=('Segoe UI', 9), relief=tk.FLAT,
                             activebackground=COLORS['bg_medium'],
                             cursor='hand2', padx=10, pady=2)
        clear_btn.pack(side=tk.RIGHT)
        
        # Results text area
        text_frame = tk.Frame(results_card, bg=COLORS['bg_medium'], padx=2, pady=2)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(text_frame, 
                                   bg=COLORS['bg_medium'], 
                                   fg=COLORS['text_primary'],
                                   font=('Consolas', 10),
                                   relief=tk.FLAT,
                                   wrap=tk.WORD,
                                   insertbackground=COLORS['text_primary'],
                                   selectbackground=COLORS['accent'],
                                   padx=10, pady=10)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(text_frame, command=self.results_text.yview,
                                bg=COLORS['bg_light'], troughcolor=COLORS['bg_medium'],
                                activebackground=COLORS['accent'])
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        
        # ==================== FOOTER ====================
        footer_frame = tk.Frame(main_frame, bg=COLORS['bg_dark'])
        footer_frame.pack(fill=tk.X, pady=(15, 0))
        
        footer_text = tk.Label(footer_frame, 
                              text="Hierarchical Pipeline: Gram - Morphology - Species  |  18 Species  |  8 Models",
                              bg=COLORS['bg_dark'], fg=COLORS['text_secondary'],
                              font=('Segoe UI', 9))
        footer_text.pack()
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def clear_results(self):
        """Clear the results text area."""
        self.results_text.delete(1.0, tk.END)
    
    def browse_path(self):
        if self.mode_var.get() == "folder":
            path = filedialog.askdirectory(title="Select Crops Folder")
        else:
            filetypes = [
                ('Image files', '*.jpg *.jpeg *.png *.tif *.tiff'),
                ('All files', '*.*')
            ]
            path = filedialog.askopenfilename(title="Select Image", filetypes=filetypes)
        
        if path:
            self.path_var.set(path)
    
    def load_models(self):
        self.analyze_btn.config(state='disabled')
        self.status_var.set("Loading models...")
        self.status_label.configure(style='Warning.TLabel')
        self.results_text.delete(1.0, tk.END)
        self.progress.start(10)
        
        thread = threading.Thread(target=self._load_models_thread)
        thread.start()
    
    def _load_models_thread(self):
        try:
            from hierarchical_pipeline import HierarchicalPipeline
            
            self.pipeline = HierarchicalPipeline({})
            
            loaded_models = []
            
            # Show loading header
            self.root.after(0, lambda: self._update_loading_status("Starting model initialization..."))
            self.root.after(0, lambda: self._append_result("=" * 60 + "\n"))
            self.root.after(0, lambda: self._append_result("  LOADING DEEP-GRAM MODELS\n"))
            self.root.after(0, lambda: self._append_result("=" * 60 + "\n\n"))
            
            # Level 1: Load Gram model
            self.root.after(0, lambda: self._update_loading_status("Loading Gram classifier..."))
            self.root.after(0, lambda: self._append_result("  [....] Loading Gram classifier...\n"))
            if os.path.exists(self.gram_model_path):
                self.pipeline.load_gram_model(self.gram_model_path)
                loaded_models.append("Gram")
                self.root.after(0, lambda: self._replace_last_line("  [OK]   Gram classifier loaded\n"))
            else:
                raise FileNotFoundError(f"Gram model not found: {self.gram_model_path}")
            
            # Level 2: Load GP Morphology model
            self.root.after(0, lambda: self._update_loading_status("Loading GP Morphology classifier..."))
            self.root.after(0, lambda: self._append_result("  [....] Loading GP Morphology classifier...\n"))
            if os.path.exists(self.gp_morphology_model_path):
                self.pipeline.load_gp_morphology_model(self.gp_morphology_model_path)
                loaded_models.append("GP-Morphology")
                self.root.after(0, lambda: self._replace_last_line("  [OK]   GP Morphology classifier loaded\n"))
            else:
                raise FileNotFoundError(f"GP Morphology model not found: {self.gp_morphology_model_path}")
            
            # Level 2: Load GN Morphology model
            self.root.after(0, lambda: self._update_loading_status("Loading GN Morphology classifier..."))
            self.root.after(0, lambda: self._append_result("  [....] Loading GN Morphology classifier...\n"))
            if os.path.exists(self.gn_morphology_model_path):
                self.pipeline.load_gn_morphology_model(self.gn_morphology_model_path)
                loaded_models.append("GN-Morphology")
                self.root.after(0, lambda: self._replace_last_line("  [OK]   GN Morphology classifier loaded\n"))
            else:
                raise FileNotFoundError(f"GN Morphology model not found: {self.gn_morphology_model_path}")
            
            # Level 3: Load species models
            self.root.after(0, lambda: self._append_result("\n  Loading Species Models:\n"))
            loaded_routes = []
            for route_name, model_path in self.species_model_paths.items():
                self.root.after(0, lambda rn=route_name: self._update_loading_status(f"Loading {rn}..."))
                self.root.after(0, lambda rn=route_name: self._append_result(f"  [....] {rn}...\n"))
                if os.path.exists(model_path):
                    self.pipeline.load_species_model(route_name, model_path)
                    loaded_routes.append(route_name)
                    self.root.after(0, lambda rn=route_name: self._replace_last_line(f"  [OK]   {rn}\n"))
                else:
                    self.root.after(0, lambda rn=route_name: self._replace_last_line(f"  [SKIP] {rn} (not found)\n"))
            
            if len(loaded_routes) == 0:
                raise FileNotFoundError("No species models found!")
            
            self.models_loaded = True
            self.root.after(0, lambda: self._models_loaded_success(loaded_models, loaded_routes))
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.root.after(0, lambda: self._models_loaded_error(error_msg))
    
    def _update_loading_status(self, message):
        """Update the status bar during loading."""
        self.status_var.set(message)
    
    def _append_result(self, text):
        """Append text to results area."""
        self.results_text.insert(tk.END, text)
        self.results_text.see(tk.END)
    
    def _replace_last_line(self, new_text):
        """Replace the last line in results area."""
        self.results_text.delete("end-2l", "end-1l")
        self.results_text.insert(tk.END, new_text)
        self.results_text.see(tk.END)
    
    def _models_loaded_success(self, loaded_models, loaded_routes):
        self.progress.stop()
        self.analyze_btn.config(state='normal')
        self.status_var.set(f"Ready  |  {len(loaded_models)} base models + {len(loaded_routes)} species models loaded")
        self.status_label.configure(style='Success.TLabel')
        
        # Show completion message
        self._append_result("\n" + "=" * 60 + "\n")
        self._append_result(f"  INITIALIZATION COMPLETE\n")
        self._append_result(f"  {len(loaded_models)} base models + {len(loaded_routes)} species models\n")
        self._append_result("=" * 60 + "\n\n")
        self._append_result("Select a folder or image above to begin analysis.\n")
    
    def _models_loaded_error(self, error):
        self.progress.stop()
        self.status_var.set("Failed to load models")
        self.status_label.configure(style='Error.TLabel')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "ERROR LOADING MODELS\n\n")
        self.results_text.insert(tk.END, error)
    
    def analyze(self):
        path = self.path_var.get()
        if not path or not os.path.exists(path):
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Please select a valid path.\n")
            return
        
        self.analyze_btn.config(state='disabled')
        self.status_var.set("Analyzing...")
        self.status_label.configure(style='Warning.TLabel')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Processing...\n\n")
        self.progress.start(10)
        
        thread = threading.Thread(target=self._analyze_thread, args=(path,))
        thread.start()
    
    def _analyze_thread(self, path):
        try:
            if self.mode_var.get() == "folder":
                # Analyze entire slide (folder of crops)
                species, confidence, summary, crop_preds = self.pipeline.predict_slide(path, aggregation='vote')
                output = self._format_slide_results(path, species, confidence, summary)
            else:
                # Analyze single image
                species, confidence, details = self.pipeline.predict_single(path)
                output = self._format_single_results(path, species, confidence, details)
            
            self.root.after(0, lambda: self._analysis_complete(output))
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.root.after(0, lambda: self._analysis_error(error_msg))
    
    def _format_single_results(self, path, species, confidence, details):
        lines = []
        lines.append("=" * 60)
        lines.append("  SINGLE BACTERIA ANALYSIS")
        lines.append("=" * 60)
        lines.append(f"\n  Image: {os.path.basename(path)}")
        
        lines.append(f"\n\n  GRAM STAINING")
        lines.append(f"  " + "-" * 40)
        lines.append(f"  Result:     {details.get('gram', 'N/A').upper()}")
        lines.append(f"  Confidence: {details.get('gram_confidence', 0):.1%}")
        
        lines.append(f"\n  MORPHOLOGY")
        lines.append(f"  " + "-" * 40)
        lines.append(f"  Result:     {details.get('morphology', 'N/A').upper()}")
        lines.append(f"  Confidence: {details.get('morphology_confidence', 0):.1%}")
        
        lines.append(f"\n  ROUTING")
        lines.append(f"  " + "-" * 40)
        lines.append(f"  Path: {details.get('route', 'N/A')}")
        
        lines.append(f"\n\n{'=' * 60}")
        lines.append(f"  PREDICTED SPECIES")
        lines.append(f"{'=' * 60}")
        lines.append(f"\n  {species.upper() if species else 'N/A'}")
        lines.append(f"  Confidence: {confidence:.1%}")
        lines.append(f"\n{'=' * 60}")
        
        return "\n".join(lines)
    
    def _format_slide_results(self, path, species, confidence, summary):
        lines = []
        lines.append("=" * 60)
        lines.append("  SLIDE ANALYSIS RESULTS")
        lines.append("=" * 60)
        lines.append(f"\n  Folder: {os.path.basename(path)}")
        lines.append(f"  Total Crops: {summary.get('total_crops', 0):,}")
        
        lines.append(f"\n\n  GRAM STAINING")
        lines.append(f"  " + "-" * 40)
        lines.append(f"  Consensus: {summary.get('slide_gram', 'N/A').upper()}")
        gram_dist = summary.get('gram_distribution', {})
        if gram_dist:
            total = sum(gram_dist.values())
            for g, count in sorted(gram_dist.items()):
                pct = 100 * count / total if total > 0 else 0
                bar = "|" * int(pct / 5) + "." * (20 - int(pct / 5))
                lines.append(f"  {g:<12} [{bar}] {pct:5.1f}%")
        
        lines.append(f"\n  MORPHOLOGY")
        lines.append(f"  " + "-" * 40)
        lines.append(f"  Consensus: {summary.get('slide_morphology', 'N/A').upper()}")
        morph_dist = summary.get('morphology_distribution', {})
        if morph_dist:
            total = sum(morph_dist.values())
            for m, count in sorted(morph_dist.items()):
                pct = 100 * count / total if total > 0 else 0
                bar = "|" * int(pct / 5) + "." * (20 - int(pct / 5))
                lines.append(f"  {m:<12} [{bar}] {pct:5.1f}%")
        
        lines.append(f"\n  ROUTING")
        lines.append(f"  " + "-" * 40)
        lines.append(f"  Path: {summary.get('slide_route', 'N/A')}")
        
        lines.append(f"\n  SPECIES DISTRIBUTION")
        lines.append(f"  " + "-" * 40)
        species_dist = summary.get('species_distribution', {})
        if species_dist:
            total = sum(species_dist.values())
            for sp, count in sorted(species_dist.items(), key=lambda x: -x[1]):
                pct = 100 * count / total if total > 0 else 0
                bar = "|" * int(pct / 5) + "." * (20 - int(pct / 5))
                # Truncate long species names
                sp_short = sp[:18] + ".." if len(sp) > 20 else sp
                lines.append(f"  {sp_short:<20} [{bar}] {pct:5.1f}%")
        
        lines.append(f"\n\n{'=' * 60}")
        lines.append(f"  PREDICTED ORGANISM")
        lines.append(f"{'=' * 60}")
        lines.append(f"\n  {species.upper() if species else 'N/A'}")
        lines.append(f"  Confidence: {confidence:.1%}")
        lines.append(f"\n{'=' * 60}")
        
        return "\n".join(lines)
    
    def _analysis_complete(self, output):
        self.progress.stop()
        self.analyze_btn.config(state='normal')
        self.status_var.set("Analysis complete")
        self.status_label.configure(style='Success.TLabel')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, output)
    
    def _analysis_error(self, error):
        self.progress.stop()
        self.analyze_btn.config(state='normal')
        self.status_var.set("Analysis failed")
        self.status_label.configure(style='Error.TLabel')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "ERROR DURING ANALYSIS\n\n")
        self.results_text.insert(tk.END, error)
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = BacteriaGUI()
    app.run()
