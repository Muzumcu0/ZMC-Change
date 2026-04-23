import joblib
import os
from datetime import datetime

def save_model(model, model_name="model"):
    """
    Eğitilmiş bir modeli, projenin ana dizinindeki 'models' klasörüne kaydeder.

    Args:
        model: Kaydedilecek eğitilmiş model nesnesi.
        model_name (str): Modelin temel adı.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '..', '..')
        model_folder = os.path.join(project_root, "models")
        os.makedirs(model_folder, exist_ok=True)

        # Dosya adını oluştururken timestamp kullanmak yerine sabit tutalım ki üzerine yazabilelim
        filename = f"{model_name}.joblib"
        filepath = os.path.join(model_folder, filename)

        joblib.dump(model, filepath)
        
        relative_path = os.path.relpath(filepath, project_root)
        print(f"Model başarıyla şu dosyaya kaydedildi: {relative_path}")

    except Exception as e:
        print(f"Model kaydedilirken bir hata oluştu: {e}")

def load_all_models():
    """
    'models' klasöründeki tüm .joblib modellerini bulur ve bir sözlük olarak yükler.
    Sözlük anahtarı, dosya adından türetilen coin sembolüdür (örn: 'BTC').

    Returns:
        dict: Coin sembollerini anahtar, yüklenmiş model nesnelerini değer olarak içeren bir sözlük.
    """
    all_models = {}
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '..', '..')
        model_folder = os.path.join(project_root, "models")

        if not os.path.exists(model_folder):
            print(f"UYARI: 'models' klasörü bulunamadı: {model_folder}")
            return all_models

        print(f"{len(os.listdir(model_folder))} adet model dosyası bulundu. Yükleniyor...")
        for filename in os.listdir(model_folder):
            if filename.endswith(".joblib"):
                try:
                    # Dosya adından coin sembolünü çıkar (örn: btc_xgboost_v1.joblib -> BTC)
                    coin_symbol = filename.split('_')[0].upper()
                    model_path = os.path.join(model_folder, filename)
                    model = joblib.load(model_path)
                    all_models[coin_symbol] = model
                    print(f" -> {filename} ({coin_symbol})")
                except Exception as e:
                    print(f"HATA: {filename} yüklenirken bir sorun oluştu: {e}")
        
        print(f"\nToplam {len(all_models)} adet model başarıyla yüklendi.")
        return all_models

    except Exception as e:
        print(f"Modeller yüklenirken genel bir hata oluştu: {e}")
        return all_models
