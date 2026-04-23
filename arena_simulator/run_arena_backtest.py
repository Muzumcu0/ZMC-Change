import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import os
import sys
from datetime import datetime, timedelta
import traceback
import json # JSON işlemleri için

# --- 1. Proje Kök Dizinini Ayarlama ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
# Arena verilerinin kaydedileceği yer (JSON dosyaları için)
ARENA_DATA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'arena_simulator')

# --- 2. Konfigürasyon ve Dosya Yolları ---
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
ARENA_MODEL_DIR = os.path.join(MODEL_DIR, 'arena_models')
FEAR_GREED_PATH = os.path.join(RAW_DATA_DIR, 'fear_greed_index.csv')

COINS = ['BTC', 'ETH', 'ADA', 'SOL', 'LTC', 'XRP', 'DOGE']
SIGNAL_MODEL_FILENAME_FORMAT = '{}_xgboost_v1.joblib'
TP_MODEL_FILENAME_FORMAT = '{}_tp_model.pkl'
SL_MODEL_FILENAME_FORMAT = '{}_sl_model.pkl'

# --- Varsayılan Parametreler ---
# (Script doğrudan çalıştırıldığında veya GET /arena/results için JSON oluştururken kullanılır)
DEFAULT_START_DATE_STR = "2025-10-17 00:00:00"
DEFAULT_END_DATE_STR = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
DEFAULT_RULE = 'Threshold_All'
DEFAULT_RISK_RATIO = 0.10
DEFAULT_THRESHOLD = 1.5
DEFAULT_CAPITAL = 10000.00

# --- Özellik Listeleri ve Sinyal Anlamları ---
MODEL_FEATURES = [
   'open', 'high', 'low', 'close', 'volume', 'fear_greed_value', 'RSI_14',
   'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'BBL_20_2.0',
   'BBM_20_2.0', 'BBU_20_2.0', 'BBP_20_2.0',
   'daily_return', 'MA7', 'MA30', 'reddit_sentiment'
]
RISK_MODEL_FEATURES = [f for f in MODEL_FEATURES if f != 'reddit_sentiment']
SIGNAL_UP = 1; SIGNAL_DOWN = 0; SIGNAL_FLAT = None

# --- 3. Veri Yükleme Fonksiyonları ---
def load_all_backtest_data(coins, start_date, end_date):
    """Tüm coinler için backtest verilerini yükler ve hazırlar."""
    all_data = {}
    print("Tüm coinler için veriler hazırlanıyor...")
    for coin in coins:
        try:
            coin_data = load_prepare_single_coin_data(coin, start_date, end_date)
            if coin_data is not None and not coin_data.empty:
                all_data[coin] = coin_data
            else:
                 print(f"Uyarı: {coin} için veri hazırlanamadı, backtest'e dahil edilmeyecek.")
        except Exception as e:
            print(f"HATA: {coin} verisi yüklenirken sorun oluştu: {e}")
    print("-" * 30)
    if not all_data: 
        print("HATA: Hiçbir coin için veri yüklenemedi.")
        return None, None, [] # Boş liste döndür
    
    # Ortak index bulma (veri olan coinler üzerinden)
    common_index = None
    loaded_coins = list(all_data.keys()) # Başarıyla yüklenen coinler
    for coin in loaded_coins:
        if common_index is None: common_index = all_data[coin].index
        else: common_index = common_index.intersection(all_data[coin].index)
        
    if common_index is None or common_index.empty: 
        print("HATA: Coin verileri arasında ortak tarih bulunamadı.")
        return None, None, []
    
    aligned_data = {}
    final_loaded_coins = [] # Ortak indexte verisi olanlar
    for coin in loaded_coins:
         try:
             aligned_data[coin] = all_data[coin].loc[common_index].copy()
             final_loaded_coins.append(coin)
         except KeyError:
              print(f"Uyarı: {coin} ortak indexte bulunamadı, atlanıyor.")
         except Exception as e:
              print(f"Hata: {coin} hizalanırken hata: {e}, atlanıyor.")

    if not final_loaded_coins: 
        print("HATA: Ortak index sonrası işlenecek coin kalmadı.")
        return None, None, []

    print(f"Ortak tarih aralığı ({len(final_loaded_coins)} coin) için veri hazırlandı: {len(common_index)} gün")
    return aligned_data, common_index, final_loaded_coins # Sadece işlenecek coin listesini döndür

def load_prepare_single_coin_data(coin, start_date, end_date):
    """Belirtilen TEK coin için veriyi yükler ve hazırlar."""
    print(f"\n--- {coin} için Veri Hazırlanıyor ---")
    market_data_path = os.path.join(RAW_DATA_DIR, f'{coin.upper()}_USDT_1d_market_data.csv')
    print(f"Market verisi yükleniyor: {market_data_path}")
    try:
        df_market = pd.read_csv(market_data_path, parse_dates=['timestamp'], index_col='timestamp')
        df_market.sort_index(inplace=True); df_market.columns = [c.lower() for c in df_market.columns]
        start_buffer = pd.Timestamp(start_date) - pd.Timedelta(days=60); df_market_buffered = df_market.loc[start_buffer:].copy()
        if df_market_buffered.empty: raise ValueError("Buffer aralığı için market verisi bulunamadı.")
        print(f"Market verisi (buffer ile) yüklendi: {len(df_market_buffered)} satır.")
    except Exception as e: print(f"HATA: {coin} market verisi okunamadı: {e}"); return None
    print(f"{coin}: Teknik göstergeler hesaplanıyor...")
    try:
        df_calc = df_market_buffered; cols_to_drop = [f for f in MODEL_FEATURES if f not in ['open','high','low','close','volume','fear_greed_value','reddit_sentiment']]
        df_calc.drop(columns=cols_to_drop, errors='ignore', inplace=True); indicators=pd.DataFrame(index=df_calc.index)
        indicators['daily_return']=df_calc['close'].pct_change(); indicators['MA7']=ta.sma(df_calc['close'],length=7); indicators['MA30']=ta.sma(df_calc['close'],length=30); indicators['RSI_14']=ta.rsi(df_calc['close'],length=14)
        macd=ta.macd(df_calc['close'],fast=12,slow=26,signal=9); bbands=ta.bbands(df_calc['close'],length=20,std=2.0)
        bbands_rename_map={c:f'BBL_20_2.0' for c in bbands.columns if c.startswith("BBL")}; bbands_rename_map.update({c:f'BBM_20_2.0' for c in bbands.columns if c.startswith("BBM")}); bbands_rename_map.update({c:f'BBU_20_2.0' for c in bbands.columns if c.startswith("BBU")}); bbands_rename_map.update({c:f'BBP_20_2.0' for c in bbands.columns if c.startswith("BBP")})
        bbands.rename(columns=bbands_rename_map,inplace=True); bb_cols_needed=['BBL_20_2.0','BBM_20_2.0','BBU_20_2.0','BBP_20_2.0']; bb_cols_available=[c for c in bb_cols_needed if c in bbands.columns]
        df_with_indicators=pd.concat([df_calc,indicators,macd,bbands[bb_cols_available]],axis=1); df_with_indicators=df_with_indicators.ffill(); print(f"{coin}: Göstergeler hesaplandı.")
    except Exception as e: print(f"HATA: {coin} gösterge hatası: {e}"); return None
    print(f"{coin}: F&G verisi yükleniyor..."); 
    try:
        df_fg=pd.read_csv(FEAR_GREED_PATH,parse_dates=['date'],index_col='date'); df_fg.sort_index(inplace=True); df_fg.index.name='timestamp'; df_fg=df_fg[['fear_greed_value']]
    except Exception as e: print(f"Uyarı: F&G okunamadı ({e}). Varsayılan (50) kullanılacak."); df_fg=pd.DataFrame(index=df_with_indicators.index,columns=['fear_greed_value']); df_fg['fear_greed_value']=50
    print(f"{coin}: Veriler birleştiriliyor..."); df_final=df_with_indicators.join(df_fg,how='left')
    if 'reddit_sentiment' not in df_final.columns: df_final['reddit_sentiment']=0
    else: df_final['reddit_sentiment']=df_final['reddit_sentiment'].fillna(0)
    df_final.replace([np.inf,-np.inf],np.nan,inplace=True); print(f"{coin}: NaN değerler dolduruluyor..."); df_final=df_final.ffill()
    df_backtest = df_final.loc[start_date:end_date].copy()
    missing_cols=[f for f in MODEL_FEATURES if f not in df_backtest.columns];
    if missing_cols: print(f"HATA: {coin} backtest verisinde özellikler eksik: {missing_cols}"); return None
    df_backtest=df_backtest[MODEL_FEATURES].copy(); df_backtest.dropna(inplace=True); print(f"{coin}: Backtest verisi: {len(df_backtest)} satır.")
    if df_backtest.empty: return None;
    if not df_backtest.index.is_unique: df_backtest=df_backtest[~df_backtest.index.duplicated(keep='first')];
    return df_backtest

# --- 4. Backtesting Motoru ---
def run_backtest_portfolio(all_data, common_index, signal_models, tp_models, sl_models, 
                           initial_capital, trade_size_ratio, active_rule, rule_threshold, 
                           coins_to_process):
    """Coin'e özel TP/SL ile portföy backtestini çalıştırır."""
    print(f"\n--- Portföy Backtest Başlatılıyor (Kural: {active_rule}) ---"); cash=initial_capital; equity=initial_capital
    positions={c:{'size':0.0,'entry_price':0.0,'tp':0.0,'sl':0.0} for c in coins_to_process}; trade_log=[]; equity_curve=pd.Series(index=common_index,dtype=float)
    for timestamp in common_index:
        current_equity=cash
        for coin,pos_info in list(positions.items()):
            if pos_info['size']==0 or coin not in all_data: continue
            try:
                current_price=all_data[coin].loc[timestamp,'close']; day_high=all_data[coin].loc[timestamp,'high']; day_low=all_data[coin].loc[timestamp,'low']
            except KeyError:
                print(f"Uyarı: {timestamp} için {coin} verisi bulunamadı, pozisyon kontrolü atlanıyor."); continue # O gün veri yoksa atla
            pnl=0; trade_closed=False; closing_price=current_price
            if pos_info['size']>0: # LONG
                if day_high>=pos_info['tp']: pnl=(pos_info['tp']-pos_info['entry_price'])*pos_info['size']; trade_closed=True; closing_price=pos_info['tp']; print(f"{timestamp} ({coin}): LONG TP ({closing_price:.2f}). PnL: ${pnl:.2f}")
                elif day_low<=pos_info['sl']: pnl=(pos_info['sl']-pos_info['entry_price'])*pos_info['size']; trade_closed=True; closing_price=pos_info['sl']; print(f"{timestamp} ({coin}): LONG SL ({closing_price:.2f}). PnL: ${pnl:.2f}")
            elif pos_info['size']<0: # SHORT
                if day_low<=pos_info['tp']: pnl=(pos_info['entry_price']-pos_info['tp'])*abs(pos_info['size']); trade_closed=True; closing_price=pos_info['tp']; print(f"{timestamp} ({coin}): SHORT TP ({closing_price:.2f}). PnL: ${pnl:.2f}")
                elif day_high>=pos_info['sl']: pnl=(pos_info['entry_price']-pos_info['sl'])*abs(pos_info['size']); trade_closed=True; closing_price=pos_info['sl']; print(f"{timestamp} ({coin}): SHORT SL ({closing_price:.2f}). PnL: ${pnl:.2f}")
            if trade_closed:
                pos_val_entry=abs(pos_info['size'])*pos_info['entry_price']; cash+=pos_val_entry+pnl
                equity_after_close=cash+sum((all_data[c].loc[timestamp,'close']-p['entry_price'])*p['size'] if p['size']>0 else (p['entry_price']-all_data[c].loc[timestamp,'close'])*abs(p['size']) for c,p in positions.items() if c!=coin and p['size']!=0 and c in all_data and timestamp in all_data[c].index)
                trade_log.append({'timestamp':timestamp,'coin':coin,'type':f"CLOSE_{'LONG' if pos_info['size']>0 else 'SHORT'}",'price':closing_price,'pnl':pnl,'cash':cash,'equity': equity_after_close})
                positions[coin]={'size':0.0,'entry_price':0.0,'tp':0.0,'sl':0.0}
            else:
                 if pos_info['size']>0: current_equity+=(current_price-pos_info['entry_price'])*pos_info['size']
                 else: current_equity+=(pos_info['entry_price']-current_price)*abs(pos_info['size'])
        
        # *** DÜZELTME 1: ESKİ SATIR SİLİNDİ ***
        # Portföy değeri (equity_curve) hesaplaması gün sonuna taşındı.
        # equity_curve.loc[timestamp]=current_equity 

        is_any_position_open_start_of_day=any(p['size']!=0 for p in positions.values())
        if not is_any_position_open_start_of_day:
            potential_trades=[]; print(f"\n{timestamp}: Açık pozisyon yok, yeni fırsatlar aranıyor...")
            for coin in coins_to_process:
                if coin not in all_data or coin not in signal_models or coin not in tp_models or coin not in sl_models: continue
                try: row=all_data[coin].loc[timestamp]; current_price=row['close']
                except KeyError: continue # O gün o coin için veri yoksa atla
                current_features_signal=pd.DataFrame([row[MODEL_FEATURES]],index=[timestamp]); current_features_risk=current_features_signal[RISK_MODEL_FEATURES]
                signal=SIGNAL_FLAT; 
                try: signal=signal_models[coin].predict(current_features_signal)[0]
                except Exception as e: print(f" Uyarı: {coin} sinyal tahmini başarısız: {e}")
                tp_pct,sl_pct,rr_ratio=0,0,0; trade_direction=None
                if signal==SIGNAL_UP:
                    try: tp_pct=max(0.001,tp_models[coin].predict(current_features_risk)[0]); sl_pct=min(-0.001,sl_models[coin].predict(current_features_risk)[0]); rr_ratio=abs(tp_pct/sl_pct) if sl_pct!=0 else float('inf'); trade_direction='LONG'
                    except Exception as e: print(f" Uyarı: {coin} UP TP/SL tahmini başarısız: {e}")
                elif signal==SIGNAL_DOWN:
                    try: tp_pct_short=min(-0.001,sl_models[coin].predict(current_features_risk)[0]); sl_pct_short=max(0.001,tp_models[coin].predict(current_features_risk)[0]); rr_ratio=abs(tp_pct_short/sl_pct_short) if sl_pct_short!=0 else float('inf'); tp_pct=tp_pct_short; sl_pct=sl_pct_short; trade_direction='SHORT'
                    except Exception as e: print(f" Uyarı: {coin} DOWN TP/SL tahmini başarısız: {e}")
                if trade_direction: potential_trades.append({'coin':coin,'direction':trade_direction,'price':current_price,'tp_pct':tp_pct,'sl_pct':sl_pct,'rr_ratio':rr_ratio}); print(f"  - {coin}: {trade_direction} Sinyali (TP: {tp_pct*100:+.2f}%, SL: {sl_pct*100:+.2f}%, R/R: {rr_ratio:.2f})")
            trades_to_open=[];
            if potential_trades:
                potential_trades.sort(key=lambda x:x['rr_ratio'],reverse=True)
                if active_rule=='Best_RiskReward_Single': trades_to_open.append(potential_trades[0]); print(f" Karar: En iyi R/R -> {trades_to_open[0]['coin']}({trades_to_open[0]['direction']})")
                elif active_rule=='Threshold_All':
                    trades_to_open=[t for t in potential_trades if t['rr_ratio']>=rule_threshold]
                    if trades_to_open: print(f" Karar: R/R>={rule_threshold} -> [{', '.join([f'{t["coin"]}({t["direction"]})' for t in trades_to_open])}]")
                    else: print(f" Karar: R/R>={rule_threshold} olan yok.")
            num_trades=len(trades_to_open)
            if num_trades>0:
                 total_investment_usd=cash*trade_size_ratio; investment_per_trade=total_investment_usd/num_trades
                 for trade in trades_to_open:
                     coin=trade['coin']; entry_price=trade['price']
                     if cash>=investment_per_trade:
                         pos_size=investment_per_trade/entry_price; cash-=investment_per_trade; tp_price=entry_price*(1+trade['tp_pct']); sl_price=entry_price*(1+trade['sl_pct'])
                         positions[coin]['entry_price']=entry_price; positions[coin]['tp']=tp_price; positions[coin]['sl']=sl_price
                         if trade['direction']=='LONG': positions[coin]['size']=pos_size
                         else: positions[coin]['size']=-pos_size
                         print(f"  * {timestamp} ({coin}): {trade['direction']} @ {entry_price:.2f} (TP:{tp_price:.2f}, SL:{sl_price:.2f}) - Yatırım: ${investment_per_trade:.2f}")
                         equity_now=cash+sum((all_data[c].loc[timestamp,'close']-p['entry_price'])*p['size'] if p['size']>0 else (p['entry_price']-all_data[c].loc[timestamp,'close'])*abs(p['size']) for c,p in positions.items() if p['size']!=0 and c in all_data and timestamp in all_data[c].index)
                         trade_log.append({'timestamp':timestamp,'coin':coin,'type':f"OPEN_{trade['direction']}",'price':entry_price,'size':positions[coin]['size'],'tp':tp_price,'sl':sl_price,'cash':cash,'equity':equity_now})
                     else: print(f"  * {timestamp} ({coin}): {trade['direction']} sinyali, yetersiz bakiye (${cash:.2f}<${investment_per_trade:.2f})"); break
        
        # --- DÜZELTME 2: YENİ KOD EKLENDİ ---
        # (Yukarıdaki 'break' ile aynı hizada olmalı)
            
        # --- GÜNLÜK DÖNGÜ SONU: EOD Portföy Değerini Hesapla ---
        # (Tüm kapanışlar ve yeni açılan pozisyonlar dahil)
        eod_equity = cash 
        for c,p in positions.items():
            if p['size'] == 0 or c not in all_data or timestamp not in all_data[c].index:
                continue
            
            try:
                # O günkü kapanış fiyatını al
                current_price = all_data[c].loc[timestamp, 'close']
            except KeyError:
                # O gün veri yoksa (tatil vs.), pozisyonun maliyetini kullan (PnL=0)
                current_price = p['entry_price'] 
            
            # Açık pozisyonların anlık (unrealized) PnL'ini nakite ekle
            if p['size'] > 0: # LONG
                unrealized_pnl = (current_price - p['entry_price']) * p['size']
                eod_equity += (p['entry_price'] * p['size']) + unrealized_pnl # Maliyet + K/Z
            elif p['size'] < 0: # SHORT
                unrealized_pnl = (p['entry_price'] - current_price) * abs(p['size'])
                eod_equity += (p['entry_price'] * abs(p['size'])) + unrealized_pnl # Maliyet + K/Z
        
        # Grafiğe (equity_curve) gün sonu toplam portföy değerini kaydet
        equity_curve.loc[timestamp] = eod_equity
        # --- EOD Hesaplaması Bitti ---

    print("--- Portföy Backtest Tamamlandı ---"); return equity_curve,trade_log

# --- 5. Ana Backtest Mantığı Fonksiyonu ---
def execute_backtest(start_date_str, end_date_str, rule, risk_ratio, threshold, capital):
    """
    Belirtilen parametrelerle backtesti çalıştırır ve JSON uyumlu sonuçları döndürür.
    """
    start_date = pd.Timestamp(start_date_str); end_date = pd.Timestamp(end_date_str)

    print("Risk (TP/SL) modelleri yükleniyor...")
    tp_models = {}; sl_models = {}; risk_models_loaded_coins = []
    for coin in COINS:
        coin_lower=coin.lower(); tp_fn=TP_MODEL_FILENAME_FORMAT.format(coin_lower); sl_fn=SL_MODEL_FILENAME_FORMAT.format(coin_lower)
        tp_path=os.path.join(ARENA_MODEL_DIR,tp_fn); sl_path=os.path.join(ARENA_MODEL_DIR,sl_fn)
        try:
            if not os.path.exists(tp_path): raise FileNotFoundError(f"TP modeli: {tp_path}")
            if not os.path.exists(sl_path): raise FileNotFoundError(f"SL modeli: {sl_path}")
            tp_models[coin]=joblib.load(tp_path); sl_models[coin]=joblib.load(sl_path); print(f" - {coin} TP/SL yüklendi."); risk_models_loaded_coins.append(coin)
        except Exception as e: print(f" UYARI: {coin} TP/SL yüklenemedi: {e}")
    if not risk_models_loaded_coins: raise Exception("HATA: Hiçbir TP/SL modeli yüklenemedi.")

    print("\nSinyal modelleri yükleniyor...")
    signal_models = {}; signal_models_loaded_coins = []
    for coin in COINS:
        coin_lower=coin.lower(); signal_fn=SIGNAL_MODEL_FILENAME_FORMAT.format(coin_lower); signal_path=os.path.join(MODEL_DIR,signal_fn)
        try:
            if not os.path.exists(signal_path): raise FileNotFoundError(f"Model: {signal_path}")
            signal_models[coin]=joblib.load(signal_path); print(f" - {coin} sinyal yüklendi."); signal_models_loaded_coins.append(coin)
        except Exception as e: print(f" UYARI: {coin} modeli yüklenemedi: {e}")
    if not signal_models_loaded_coins: raise Exception("HATA: Hiçbir sinyal modeli yüklenemedi.")

    coins_to_process = list(set(risk_models_loaded_coins) & set(signal_models_loaded_coins))
    if not coins_to_process: raise Exception("HATA: Ortak coin bulunamadı.")
    print(f"\nİşlenecek coinler ({len(coins_to_process)}): {', '.join(coins_to_process)}")

    all_data, common_index, loaded_coins = load_all_backtest_data(coins_to_process, start_date, end_date)
    if all_data is None: raise Exception("HATA: Backtest verisi hazırlanamadı.")
    
    coins_final = [c for c in coins_to_process if c in loaded_coins]
    if not coins_final: raise Exception("HATA: Verisi yüklenebilen işlenecek coin bulunamadı.")
    print(f"Verisi olan işlenecek coinler ({len(coins_final)}): {', '.join(coins_final)}")

    equity_curve, trade_log = run_backtest_portfolio(
        all_data, common_index, signal_models, tp_models, sl_models, 
        capital, risk_ratio, rule, threshold, coins_final
    )

    # --- Sonuçları Hesapla (Daha Güvenli) ---
    print("Sonuçlar hesaplanıyor...")
    final_equity = equity_curve.iloc[-1] if not equity_curve.empty else capital
    total_return_pct = ((final_equity / capital) - 1) * 100 if capital > 0 else 0.0
    trade_log_df = pd.DataFrame(trade_log)
    opened_trades_count = len(trade_log_df[trade_log_df['type'].str.contains('OPEN')])
    closed_trades = trade_log_df[trade_log_df['type'].str.contains("CLOSE")]
    win_rate, avg_win, avg_loss, profit_factor = 0.0, 0.0, 0.0, 0.0
    if not closed_trades.empty:
        winning_trades = closed_trades[closed_trades['pnl'] > 0]
        losing_trades = closed_trades[closed_trades['pnl'] <= 0]
        win_rate = (len(winning_trades) / len(closed_trades)) * 100 if len(closed_trades) > 0 else 0.0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0.0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0.0
        total_profit = winning_trades['pnl'].sum()
        total_loss = abs(losing_trades['pnl'].sum())
        if total_loss > 0: profit_factor = total_profit / total_loss
        elif total_profit > 0 and total_loss == 0: profit_factor = float('inf')
        else: profit_factor = 0.0
    
    # Sonuçları Dictionary olarak hazırla
    summary_data = {
        "start_capital": capital, "final_equity": final_equity, "total_return_pct": total_return_pct,
        "opened_trades": opened_trades_count, "closed_trades": len(closed_trades),
        "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss, "profit_factor": profit_factor
    }
    equity_curve_dict_strkey = {k.strftime('%Y-%m-%d'): v for k, v in equity_curve.dropna().to_dict().items()}
    trade_log_list = trade_log_df.astype({'timestamp': str}).to_dict('records') if not trade_log_df.empty else []

    # --- YENİ: JSON'a göndermeden önce NaN/Inf değerlerini temizle ---
    print("Sonuçlardaki NaN/Inf değerleri JSON uyumlu hale getiriliyor (None)...")
    
    def clean_value(value):
        """ NaN veya Inf ise None döndür, değilse değeri döndür """
        # np.float_ kaldırıldı
        if isinstance(value, (float, np.float64)) and (np.isnan(value) or np.isinf(value)):
            return None # None, JSON'da 'null' olur
        return value

    for key, value in summary_data.items():
        summary_data[key] = clean_value(value)
    valid_equity_keys = list(equity_curve_dict_strkey.keys())
    for key in valid_equity_keys:
         value = equity_curve_dict_strkey.get(key) # .get() ile daha güvenli
         equity_curve_dict_strkey[key] = clean_value(value)
    for trade in trade_log_list:
        for key, value in trade.items():
             trade[key] = clean_value(value)
    # --- NaN Temizleme Bitti ---

    print("Temizlenmiş sonuçlar API için döndürülüyor.")
    return {
        "summary": summary_data,
        "equity_curve": equity_curve_dict_strkey,
        "trade_log": trade_log_list
    }

# --- 6. Ana Çalıştırıcı (Varsayılan çalıştırma) ---
if __name__ == "__main__":
    print("*"*50); print(" Arena Backtest Çalıştırılıyor (Varsayılan Parametreler) "); print("*"*50)
    try:
        results = execute_backtest(
            start_date_str=DEFAULT_START_DATE_STR, end_date_str=DEFAULT_END_DATE_STR, 
            rule=DEFAULT_RULE, risk_ratio=DEFAULT_RISK_RATIO, 
            threshold=DEFAULT_THRESHOLD, capital=DEFAULT_CAPITAL
        )
        print("\nSonuçlar frontend için JSON olarak kaydediliyor...")
        equity_json_path=os.path.join(ARENA_DATA_OUTPUT_DIR,'equity_curve.json')
        summary_json_path=os.path.join(ARENA_DATA_OUTPUT_DIR,'summary.json')
        trade_log_json_path=os.path.join(ARENA_DATA_OUTPUT_DIR,'trade_log.json')
        with open(equity_json_path,'w', encoding='utf-8') as f: json.dump(results["equity_curve"],f,indent=4) # encoding eklendi
        print(f"Portföy değeri kaydedildi: {equity_json_path}")
        with open(summary_json_path,'w', encoding='utf-8') as f: json.dump(results["summary"],f,indent=4) # encoding eklendi
        print(f"Özet veriler kaydedildi: {summary_json_path}")
        with open(trade_log_json_path,'w', encoding='utf-8') as f: json.dump(results["trade_log"],f,indent=4) # encoding eklendi
        print(f"İşlem logu kaydedildi: {trade_log_json_path}")
        summary=results["summary"]
        print("\n--- Portföy Backtest Sonuçları ---")
        # Özet yazdırma kısmında None kontrolü ekleyelim
        print(f"Başlangıç Sermayesi: ${summary.get('start_capital', 0):.2f}")
        print(f"Bitiş Sermayesi:     ${summary.get('final_equity', 0):.2f}")
        print(f"Toplam Getiri:       {summary.get('total_return_pct', 0):.2f}%")
        print(f"\nToplam İşlem Sayısı (Açılış): {summary.get('opened_trades', 0)}")
        print(f"Kapatılan İşlem Sayısı:    {summary.get('closed_trades', 0)}")
        if summary.get('closed_trades', 0) > 0: 
            print(f"Kazanma Oranı:           {summary.get('win_rate', 0):.2f}%")
            print(f"Ortalama Kazançlı İşlem: ${summary.get('avg_win', 0):.2f}")
            print(f"Ortalama Kaybeden İşlem: ${summary.get('avg_loss', 0):.2f}")
            print(f"Toplam Kar / Toplam Zarar (Profit Factor): {summary.get('profit_factor', 0):.2f}")
        else: print("\nHiç işlem kapanmadı.")
    except Exception as e: print(f"\nKRİTİK HATA: Backtest çalıştırılırken hata oluştu: {e}"); traceback.print_exc()
    print("\nScript tamamlandı.")