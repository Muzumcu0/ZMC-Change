import praw
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime
import time # Hata düzeltmesi için bu satır eklendi

def initialize_reddit():
    """
    .env dosyasından API bilgilerini yükler ve Reddit API'sine bağlanır.

    Returns:
        praw.Reddit: Reddit API'sine bağlanmış PRAW nesnesi.
    """
    load_dotenv() # .env dosyasındaki değişkenleri yükler

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")

    if not all([client_id, client_secret, user_agent]):
        raise ValueError("Lütfen .env dosyasında REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET ve REDDIT_USER_AGENT değişkenlerini tanımlayın.")

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )
    return reddit

def fetch_reddit_posts(reddit, subreddit_name, limit=100):
    """
    Belirtilen subreddit'ten gönderileri çeker.

    Args:
        reddit (praw.Reddit): PRAW nesnesi.
        subreddit_name (str): Veri çekilecek subreddit'in adı.
        limit (int): Çekilecek maksimum gönderi sayısı.

    Returns:
        list: Gönderi bilgilerini içeren bir sözlük listesi.
    """
    print(f"-> '{subreddit_name}' subreddit'inden son {limit} gönderi çekiliyor...")
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []
    
    # Son 24 saatin en popüler gönderilerini çekiyoruz ('hot', 'new', 'top' olabilir)
    # 'top' ile son günün en popülerlerini alalım
    for submission in subreddit.top(time_filter='day', limit=limit):
        posts_data.append({
            'created_utc': datetime.utcfromtimestamp(submission.created_utc),
            'title': submission.title,
            'score': submission.score,
            'num_comments': submission.num_comments,
            'url': submission.url
        })
    print(f"-> {len(posts_data)} adet gönderi bulundu.")
    return posts_data

if __name__ == "__main__":
    # Hangi coin için hangi subreddit'lerin taranacağını belirliyoruz.
    # Bir coin için birden fazla subreddit de eklenebilir.
    SUBREDDIT_MAP = {
        'BTC': ['Bitcoin', 'CryptoCurrency'],
        'ETH': ['ethereum', 'ethtrader'],
        'DOGE': ['dogecoin'],
        'LTC': ['litecoin'],
        'XRP': ['ripple'],
        'SOL': ['solana'],
        'ADA': ['cardano']
    }

    output_folder = os.path.join('data', 'raw')
    os.makedirs(output_folder, exist_ok=True)

    try:
        reddit = initialize_reddit()

        for coin, subreddits in SUBREDDIT_MAP.items():
            all_posts_for_coin = []
            print(f"--------------------------------------------------")
            print(f"İşlem başlıyor: {coin}")
            
            for sub in subreddits:
                # Dosya adını oluştur ve var olup olmadığını kontrol et
                # Şimdilik her çalıştırmada yeniden çekmesi için bu kontrolü kapalı tutalım
                # output_path = os.path.join(output_folder, f"{coin}_{sub}_reddit_data.csv")
                
                posts = fetch_reddit_posts(reddit, sub, limit=100)
                all_posts_for_coin.extend(posts)
                time.sleep(1) # API limitlerine takılmamak için bekle

            if all_posts_for_coin:
                df = pd.DataFrame(all_posts_for_coin)
                # Aynı gönderi birden fazla subreddit'te paylaşıldıysa diye kopyaları temizleyelim
                df.drop_duplicates(subset=['title'], inplace=True)
                
                final_output_path = os.path.join(output_folder, f"{coin}_reddit_data.csv")
                df.to_csv(final_output_path, index=False)
                print(f"BAŞARILI: {coin} için toplam {len(df)} gönderi şu dosyaya kaydedildi -> {final_output_path}")

        print("--------------------------------------------------")
        print("Tüm coinler için Reddit veri çekme işlemi başarıyla tamamlandı.")

    except Exception as e:
        print(f"BAŞARISIZ: İşlem sırasında bir hata oluştu: {e}")

