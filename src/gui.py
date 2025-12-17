import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import subprocess
import shutil
import threading
import numpy as np
import os
import time

from compression_manager import CompressionManager
from compressed_data import CompressedData

# --- GLOBALS (Réduites au strict minimum) ---
current_file_path = None
original_image_pil = None
compressed_archive = None
last_dir = os.path.expanduser("~")
current_base_name = "compressed"


# --- COMPONENT: IMAGE VIEWER CLASS ---
class ImageViewer(tk.Canvas):
    """
    Composant dédié à l'affichage d'images redimensionnables.
    Remplace les variables globales et la logique de Label.
    """

    def __init__(self, master, **kwargs):
        # Fond gris clair par défaut pour distinguer la zone
        super().__init__(master, highlightthickness=0, **kwargs)
        self.original_image = None
        self.photo_ref = None  # Référence pour éviter le Garbage Collector
        self._resize_timer = None

        # Binding interne : le composant gère son propre redimensionnement
        self.bind("<Configure>", self._on_resize)

    def set_image(self, pil_img):
        """Définit la nouvelle image à afficher."""
        self.original_image = pil_img
        self._update_view()

    def clear(self):
        """Efface l'image."""
        self.original_image = None
        self.delete("all")

    def _on_resize(self, event):
        """Gestionnaire d'événement avec Debounce (anti-lag)."""
        if self._resize_timer:
            self.after_cancel(self._resize_timer)
        # On attend 50ms de pause avant de redessiner (LANCZOS est lourd)
        self._resize_timer = self.after(50, self._update_view)

    def _update_view(self):
        """Redessine l'image centrée et redimensionnée."""
        if not self.original_image:
            return

        # Dimensions du Canvas
        w_canvas = self.winfo_width()
        h_canvas = self.winfo_height()

        if w_canvas < 10 or h_canvas < 10:
            return

        # Calcul du ratio pour "Fit" (contenir l'image sans la couper)
        w_img, h_img = self.original_image.size
        ratio = min(w_canvas / w_img, h_canvas / h_img)

        new_w = int(w_img * ratio)
        new_h = int(h_img * ratio)

        # Redimensionnement haute qualité
        # Note: Si l'image est très petite, on évite de l'agrandir (optionnel)
        # Ici on permet l'agrandissement pour remplir l'espace.
        img_resized = self.original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        self.photo_ref = ImageTk.PhotoImage(img_resized)

        # Dessin sur le Canvas
        self.delete("all")
        # On place l'image pile au centre du Canvas
        self.create_image(w_canvas // 2, h_canvas // 2, image=self.photo_ref, anchor="center")


# --- HELPER FUNCTIONS FOR DIALOGS ---

def ask_open_filename_custom():
    global last_dir
    file_types_tk = [
        ("All Supported Files", "*.frac *.png *.jpg *.jpeg *.bmp"),
        ("Fractal Compressed", "*.frac"),
        ("Images", "*.png *.jpg *.jpeg *.bmp")
    ]
    if shutil.which("kdialog"):
        try:
            k_filter = "All Supported (*.frac *.png *.jpg *.jpeg)|Fractal Files (*.frac)|Images (*.png *.jpg *.jpeg)"
            cmd = ["kdialog", "--title", "Open File", "--getopenfilename", last_dir, k_filter]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                path = res.stdout.strip()
                if path: return path
            return None
        except:
            pass
    return filedialog.askopenfilename(
        parent=root, title="Open File", initialdir=last_dir, filetypes=file_types_tk
    )


def ask_save_filename_custom(default_name):
    global last_dir
    full_default_path = os.path.join(last_dir, default_name)
    if shutil.which("kdialog"):
        try:
            cmd = ["kdialog", "--title", "Save Compressed File", "--getsavefilename", full_default_path,
                   "Fractal Compressed (*.frac)"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                path = res.stdout.strip()
                if path:
                    if not path.endswith(".frac"): path += ".frac"
                    return path
            return None
        except:
            return None
    return filedialog.asksaveasfilename(
        parent=root, title="Save Compressed File", initialdir=last_dir, initialfile=default_name,
        defaultextension=".frac", filetypes=[("Fractal Compressed", "*.frac")]
    )


# --- CORE LOGIC ---

def open_image():
    global current_file_path, original_image_pil, compressed_archive, last_dir, current_base_name

    file_path = ask_open_filename_custom()

    if file_path:
        current_file_path = file_path
        last_dir = os.path.dirname(file_path)
        filename_no_ext = os.path.splitext(os.path.basename(file_path))[0]
        current_base_name = filename_no_ext + ".frac"
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".frac":
            status_var.set(f"Loading archive: {os.path.basename(file_path)}...")
            progress_bar.configure(mode='indeterminate')
            progress_bar.start(10)

            def load_frac_thread():
                global compressed_archive, original_image_pil
                try:
                    archive = CompressedData.load_from_file(file_path)
                    if not archive: raise ValueError("Failed to load archive data.")
                    compressed_archive = archive

                    root.after(0, lambda: status_var.set("Decompressing..."))
                    img = CompressionManager.decompress(archive, verbose=True)
                    original_image_pil = img

                    def update_ui_frac():
                        # UTILISATION DE LA CLASSE IMAGEVIEWER
                        image_viewer.set_image(img)

                        meta = archive.get_frame(0)['meta']
                        mode = meta.get('original_mode', '?')
                        algo = meta.get('quantization', 'float')

                        status_var.set(f"Opened .frac | {mode} | {algo}")
                        progress_bar.stop()
                        progress_bar.configure(mode='determinate')
                        progress_var.set(0)
                        btn_compress.config(state="normal")
                        btn_save.config(state="normal")

                    root.after(0, update_ui_frac)

                except Exception as e:
                    print(f"Error: {e}")
                    root.after(0, lambda: messagebox.showerror("Error", f"Load Error:\n{e}"))

                    def reset_bar_error():
                        progress_bar.stop()
                        progress_bar.configure(mode='determinate')
                        progress_var.set(0)

                    root.after(0, reset_bar_error)
                    root.after(0, lambda: status_var.set("Error."))

            threading.Thread(target=load_frac_thread, daemon=True).start()

        else:
            try:
                img = Image.open(file_path)
                original_image_pil = img
                compressed_archive = None

                # UTILISATION DE LA CLASSE IMAGEVIEWER
                image_viewer.set_image(img)

                status_var.set(f"Loaded: {os.path.basename(file_path)} ({img.size[0]}x{img.size[1]})")
                btn_compress.config(state="normal")
                btn_save.config(state="disabled")
                progress_var.set(0)
            except Exception as e:
                messagebox.showerror("Error", f"Could not open image:\n{e}")


def save_archive():
    global current_base_name
    if compressed_archive:
        file_path = ask_save_filename_custom(current_base_name)
        if file_path:
            global last_dir
            last_dir = os.path.dirname(file_path)

            def save_thread():
                root.after(0, lambda: status_var.set("Saving..."))
                try:
                    success = compressed_archive.save_to_file(file_path)
                    if success:
                        root.after(0, lambda: messagebox.showinfo("Success", f"Saved: {os.path.basename(file_path)}"))
                        root.after(0, lambda: status_var.set("File Saved."))
                    else:
                        root.after(0, lambda: messagebox.showerror("Error", "Save Failed."))
                except Exception as e:
                    root.after(0, lambda: messagebox.showerror("Error", f"Save Error: {e}"))

            threading.Thread(target=save_thread, daemon=True).start()


def update_progress_ui(percent):
    root.after(0, lambda: progress_var.set(percent))


def compression_task():
    global compressed_archive
    try:
        root.after(0, lambda: status_var.set("Compressing..."))
        root.after(0, lambda: btn_compress.config(state="disabled"))
        root.after(0, lambda: btn_open.config(state="disabled"))

        start_time = time.time()
        archive = CompressionManager.compress(original_image_pil, verbose=True, progress_callback=update_progress_ui)
        duration = time.time() - start_time
        compressed_archive = archive

        root.after(0, lambda: status_var.set("Decompressing preview..."))
        rec_pil = CompressionManager.decompress(archive, verbose=True)

        arr_orig = np.array(original_image_pil.convert('RGB'))
        arr_recon = np.array(rec_pil.convert('RGB'))
        mse = np.mean((arr_orig - arr_recon) ** 2)
        psnr = 100 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

        quality_label = "Poor"
        if psnr >= 40:
            quality_label = "Excellent"
        elif psnr >= 30:
            quality_label = "Good"
        elif psnr >= 25:
            quality_label = "Average"

        def finish():
            # UTILISATION DE LA CLASSE IMAGEVIEWER
            image_viewer.set_image(rec_pil)

            status_msg = f"Done! PSNR: {psnr:.2f}dB ({quality_label}) | Time: {duration:.2f}s"
            status_var.set(status_msg)
            btn_compress.config(state="normal")
            btn_open.config(state="normal")
            btn_save.config(state="normal")
            full_msg = f"Compression Finished Successfully.\n\nTime Taken: {duration:.2f} seconds\nQuality (PSNR): {psnr:.2f} dB ({quality_label})"
            messagebox.showinfo("Done", full_msg)

        root.after(0, finish)

    except Exception as e:
        print(f"Error: {e}")
        root.after(0, lambda: status_var.set(f"Error: {e}"))
        root.after(0, lambda: btn_compress.config(state="normal"))
        root.after(0, lambda: btn_open.config(state="normal"))


def start_compression():
    threading.Thread(target=compression_task, daemon=True).start()


# --- GUI SETUP ---
root = ttk.Window(themename="cosmo")
root.title("Fractal Compressor Pro")
root.geometry("900x700")

toolbar = ttk.Frame(root, padding=10)
toolbar.pack(side=TOP, fill=X)

btn_open = ttk.Button(toolbar, text="Open", bootstyle=(PRIMARY, OUTLINE), command=open_image)
btn_open.pack(side=LEFT, padx=5)

btn_compress = ttk.Button(toolbar, text="Compress", bootstyle=(SUCCESS), state="disabled", command=start_compression)
btn_compress.pack(side=LEFT, padx=5)

btn_save = ttk.Button(toolbar, text="Save", bootstyle=(INFO, OUTLINE), state="disabled", command=save_archive)
btn_save.pack(side=LEFT, padx=5)

progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(toolbar, variable=progress_var, maximum=100, bootstyle=(SUCCESS, STRIPED), length=300)
progress_bar.pack(side=RIGHT, padx=10, fill=X)

ttk.Separator(root, orient=HORIZONTAL).pack(fill=X)

main_frame = ttk.Frame(root)
main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

# REMPLACEMENT DU LABEL PAR NOTRE COMPOSANT CANVAS
image_viewer = ImageViewer(main_frame, bg='#f0f0f0')
image_viewer.pack(fill=BOTH, expand=True)

# Placeholder text via Canvas (plus propre)
image_viewer.create_text(
    400, 300,
    text="No image loaded",
    fill="gray",
    font=("Helvetica", 14),
    tags="placeholder"
)


# Astuce: On bind le resize du Canvas pour centrer le placeholder s'il n'y a pas d'image
def center_placeholder(event):
    if not image_viewer.original_image:
        image_viewer.delete("placeholder")
        image_viewer.create_text(
            event.width // 2, event.height // 2,
            text="No image loaded",
            fill="gray",
            font=("Helvetica", 14),
            tags="placeholder"
        )


image_viewer.bind("<Configure>", center_placeholder, add="+")

status_var = tk.StringVar(value="Ready")
ttk.Label(root, textvariable=status_var, bootstyle=INVERSE, padding=5).pack(side=BOTTOM, fill=X)

root.mainloop()