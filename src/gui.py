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

from compression_manager import CompressionManager
from compressed_data import CompressedData

# Globals
current_file_path = None
original_image_pil = None
compressed_archive = None


def open_image():
    global current_file_path, original_image_pil, compressed_archive
    file_path = ""

    # Configuration des filtres pour inclure .frac
    file_types = [
        ("All Supported Files", "*.frac *.png *.jpg *.jpeg *.bmp"),
        ("Fractal Compressed", "*.frac"),
        ("Images", "*.png *.jpg *.jpeg *.bmp")
    ]

    # 1. KDialog (Linux KDE)
    if shutil.which("kdialog"):
        try:
            # Construction de la chaîne de filtre pour kdialog
            # Format: "Description (*.ext1 *.ext2)|Description2 (*.ext3)"
            k_filter = "All Supported (*.frac *.png *.jpg *.jpeg)|Fractal Files (*.frac)|Images (*.png *.jpg *.jpeg)"
            cmd = ["kdialog", "--title", "Open File", "--getopenfilename", ".", k_filter]
            res = subprocess.run(cmd, capture_output=True, text=True)
            file_path = res.stdout.strip()
        except:
            pass

    # 2. Fallback Tkinter
    if not file_path:
        file_path = filedialog.askopenfilename(
            parent=root,
            title="Open File",
            filetypes=file_types
        )

    if file_path:
        current_file_path = file_path
        ext = os.path.splitext(file_path)[1].lower()

        # --- CAS 1 : Fichier Compressé (.frac) ---
        if ext == ".frac":
            status_var.set(f"Loading archive: {os.path.basename(file_path)}...")
            progress_bar.start(10)  # Animation d'attente

            def load_frac_thread():
                global compressed_archive, original_image_pil
                try:
                    # 1. Chargement disque
                    archive = CompressedData.load_from_file(file_path)
                    if not archive:
                        raise ValueError("Failed to load archive data.")

                    compressed_archive = archive  # On stocke l'archive chargée

                    # 2. Décompression
                    root.after(0, lambda: status_var.set("Decompressing..."))
                    img = CompressionManager.decompress(archive, verbose=True)
                    original_image_pil = img  # L'image chargée devient la "source" visible

                    # 3. Mise à jour UI
                    def update_ui_frac():
                        display_image(img)

                        # Infos sur l'archive
                        meta = archive.get_frame(0)['meta']
                        mode = meta.get('original_mode', '?')
                        sub = meta.get('subsampling', 'None')
                        algo = meta.get('quantization', 'float')

                        status_var.set(f"Opened .frac | Mode: {mode} {sub} | Algo: {algo}")
                        progress_bar.stop()

                        # On active tout (on peut re-compresser le résultat ou le re-sauvegarder)
                        btn_compress.config(state="normal")
                        btn_save.config(state="normal")

                    root.after(0, update_ui_frac)

                except Exception as e:
                    print(f"Error loading frac: {e}")
                    root.after(0, lambda: messagebox.showerror("Error", f"Could not load .frac file:\n{e}"))
                    root.after(0, lambda: progress_bar.stop())
                    root.after(0, lambda: status_var.set("Error loading file."))

            threading.Thread(target=load_frac_thread, daemon=True).start()

        # --- CAS 2 : Image Standard ---
        else:
            try:
                img = Image.open(file_path)
                original_image_pil = img
                compressed_archive = None  # On reset l'archive car c'est une nouvelle image brute

                display_image(img)

                status_var.set(f"Loaded: {os.path.basename(file_path)} ({img.size[0]}x{img.size[1]})")
                btn_compress.config(state="normal")
                btn_save.config(state="disabled")  # Pas d'archive à sauver pour l'instant
                progress_var.set(0)
            except Exception as e:
                messagebox.showerror("Error", f"Could not open image:\n{e}")


def display_image(pil_img):
    """Helper pour afficher l'image dans le label Tkinter"""
    disp = pil_img.copy()
    disp.thumbnail((800, 600))
    photo = ImageTk.PhotoImage(disp)
    image_label.config(image=photo)
    image_label.image = photo  # Garder la ref !


def save_archive():
    """Sauvegarde avec gestionnaire natif si possible."""
    if compressed_archive:
        file_path = ""
        if shutil.which("kdialog"):
            try:
                cmd = ["kdialog", "--title", "Save Compressed File", "--getsavefilename", "compressed.frac",
                       "Fractal Compressed (*.frac)"]
                res = subprocess.run(cmd, capture_output=True, text=True)
                file_path = res.stdout.strip()
            except:
                pass

        if not file_path:
            file_path = filedialog.asksaveasfilename(parent=root, title="Save Compressed File",
                                                     defaultextension=".frac",
                                                     filetypes=[("Fractal Compressed", "*.frac")])

        if file_path:
            # On lance la sauvegarde dans un thread aussi (LZMA peut être un peu lent sur gros fichiers)
            def save_thread():
                root.after(0, lambda: status_var.set("Saving..."))
                try:
                    success = compressed_archive.save_to_file(file_path)
                    if success:
                        root.after(0, lambda: messagebox.showinfo("Success", f"Saved to {file_path}"))
                        root.after(0, lambda: status_var.set("File Saved."))
                    else:
                        root.after(0, lambda: messagebox.showerror("Error", "Failed to save file."))
                        root.after(0, lambda: status_var.set("Save Failed."))
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

        archive = CompressionManager.compress(
            original_image_pil,
            verbose=True,
            progress_callback=update_progress_ui
        )
        compressed_archive = archive

        root.after(0, lambda: status_var.set("Decompressing preview..."))
        rec_pil = CompressionManager.decompress(archive, verbose=True)

        # Calcul PSNR
        arr_orig = np.array(original_image_pil.convert('RGB'))
        arr_recon = np.array(rec_pil.convert('RGB'))
        mse = np.mean((arr_orig - arr_recon) ** 2)
        psnr = 100 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

        def finish():
            display_image(rec_pil)
            status_var.set(f"Done! PSNR: {psnr:.2f}dB")
            btn_compress.config(state="normal")
            btn_open.config(state="normal")
            btn_save.config(state="normal")
            messagebox.showinfo("Done", f"Compression Finished.\nPSNR: {psnr:.2f}dB")

        root.after(0, finish)

    except Exception as e:
        print(f"Error: {e}")
        root.after(0, lambda: status_var.set(f"Error: {e}"))
        root.after(0, lambda: btn_compress.config(state="normal"))
        root.after(0, lambda: btn_open.config(state="normal"))


def start_compression():
    threading.Thread(target=compression_task, daemon=True).start()


# --- GUI ---
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
image_label = ttk.Label(main_frame, text="No image loaded", anchor="center")
image_label.pack(fill=BOTH, expand=True)

status_var = tk.StringVar(value="Ready")
ttk.Label(root, textvariable=status_var, bootstyle=INVERSE, padding=5).pack(side=BOTTOM, fill=X)

root.mainloop()