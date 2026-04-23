import tkinter as tk
from tkinter import ttk, font
import uvicorn
import threading
import sys
import os
import webbrowser # Linki açmak için eklendi

# --- Proje Yolu Ayarları ---
# PyInstaller'ın .exe içindeki yolları doğru bulması için bu önemlidir.
if getattr(sys, 'frozen', False):
    # .exe olarak çalışıyorsa
    PROJECT_ROOT = sys._MEIPASS
else:
    # .py olarak çalışıyorsa
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# API'nin (src) ve diğer modüllerin (arena_simulator) yolunu ekle
# Böylece "from src.api.main import app" çalışabilir
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
sys.path.append(PROJECT_ROOT)

try:
    # Ana FastAPI uygulamanızı import edin
    from src.api.main import app
except ImportError as e:
    # .exe paketlemesinde bu olabilir, alternatif yolu deneyin
    try:
        sys.path.append(os.path.join(PROJECT_ROOT))
        from src.api.main import app
    except ImportError:
        # Hata durumunda kullanıcıyı bilgilendir
        import traceback
        error_msg = f"Kritik Hata: 'src.api.main' import edilemedi.\n{e}\n{traceback.format_exc()}"
        print(error_msg)
        # GUI çalışmadan önce hata ver
        tk.Tk().withdraw() # Ana pencereyi gizle
        tk.messagebox.showerror("Launcher Hatası", error_msg)
        sys.exit(1)


# --- Uvicorn Sunucu Sınıfı ---
class APILauncher:
    
    def __init__(self, root):
        self.root = root
        self.server = None
        self.server_thread = None
        self.api_url = "http://127.0.0.1:8000"

        self.setup_ui()
        
    def setup_ui(self):
        self.root.title("ZMC-Change API Launcher")
        self.root.geometry("400x350")
        self.root.resizable(False, False)
        
        # Stil ayarları
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#252637')
        style.configure('TLabel', background='#252637', foreground='#ffffff', font=('Inter', 10))
        style.configure('Title.TLabel', font=('Inter', 16, 'bold'))
        style.configure('Status.TLabel', font=('Inter', 12, 'bold'))
        style.configure('TButton', font=('Inter', 12, 'bold'), padding=10)
        style.map('TButton',
            foreground=[('active', '#ffffff'), ('!disabled', '#ffffff')],
            background=[('active', '#8b5cf6'), ('!disabled', '#00d4ff')])
        
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Başlık
        title_label = ttk.Label(main_frame, text="ZMC-CHANGE API", style='Title.TLabel')
        title_label.pack(pady=(0, 20))

        # Durum
        self.status_label = ttk.Label(main_frame, text="Deactivated", style='Status.TLabel', foreground='#ef4444')
        self.status_label.pack(pady=5)

        # Link
        link_label = ttk.Label(main_frame, text="API Address:")
        link_label.pack(pady=(15, 0))
        
        self.link_entry = ttk.Entry(main_frame, width=30, font=('Inter', 10))
        self.link_entry.insert(0, self.api_url)
        self.link_entry.config(state='readonly')
        self.link_entry.pack(pady=5)
        
        # Linki açma butonu
        open_link_btn = ttk.Button(main_frame, text="Open Link", command=self.open_link)
        open_link_btn.pack(pady=5)

        # Başlat/Durdur Butonu
        self.toggle_button = ttk.Button(main_frame, text="Activate API", command=self.toggle_api, width=20)
        self.toggle_button.pack(pady=(20, 0))

    def open_link(self):
        webbrowser.open_new(self.api_url)

    def toggle_api(self):
        if self.server and self.server.started:
            self.stop_server()
        else:
            self.start_server()
            
    def start_server(self):
        # Sunucuyu bir thread içinde başlat ki arayüz donmasın
        
        # Uvicorn ayarları
        config = uvicorn.Config(
            app=app, # Import ettiğimiz FastAPI uygulaması
            host="127.0.0.1",
            port=8000,
            log_level="info"
        )
        self.server = uvicorn.Server(config)
        
        # daemon=True, ana program (GUI) kapandığında thread'in de kapanmasını sağlar
        self.server_thread = threading.Thread(target=self.server.run, daemon=True)
        self.server_thread.start()
        
        self.update_ui_status(active=True)
        
    def stop_server(self):
        if self.server:
            # Uvicorn'a "artık kapan" sinyalini gönder
            self.server.should_exit = True
            
            # Thread'in işini bitirmesini bekle
            # self.server_thread.join() # Bu bazen donmaya neden olabilir, daemon thread'e güvenmek daha iyi
            
            self.server = None
            self.server_thread = None
            
            self.update_ui_status(active=False)

    def update_ui_status(self, active):
        if active:
            self.status_label.config(text="Active", foreground='#10b981') # Yeşil
            self.toggle_button.config(text="Deactivate API")
        else:
            self.status_label.config(text="Deactivated", foreground='#ef4444') # Kırmızı
            self.toggle_button.config(text="Activate API")

    def on_close(self):
        # Pencereyi kapatırken sunucuyu da durdur
        print("Launcher kapatılıyor, sunucu durduruluyor...")
        self.stop_server()
        self.root.destroy()

# --- Uygulamayı Başlat ---
if __name__ == "__main__":
    # Uvicorn'un multiprocessing kullandığı durumlarda 
    # PyInstaller için bu satır gereklidir.
    import multiprocessing
    multiprocessing.freeze_support() 

    root = tk.Tk()
    app_launcher = APILauncher(root)
    
    # Pencere kapatma (X) butonuna basıldığında on_close fonksiyonunu çağır
    root.protocol("WM_DELETE_WINDOW", app_launcher.on_close)
    
    root.mainloop()