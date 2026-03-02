import streamlit as st
import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import datetime
import io
import requests
import time
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pykrx import stock
from pykrx import stock as krx_stock

# 페이지 설정 (넓은 레이아웃 사용)
st.set_page_config(page_title="Stock Alpha Tracker", layout="wide")

# ---------------------------------------------------------
# 상수 및 헬퍼 함수
# ---------------------------------------------------------
def to_yahoo_ticker(code, market):
    """종목코드와 시장 정보를 야후 파이낸스 티커 형식으로 변환합니다."""
    return f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ"

def normalize_high_proximity_score(hp_series):
    """
    신고가 근접도(High Proximity)를 시그모이드 함수를 사용하여 0에서 100 사이로 정규화합니다.
    85% 이하는 0점, 100% 이상은 100점, 그 사이는 시그모이드 곡선을 따릅니다.
    """
    normalized_hp_series = hp_series.copy()

    # 점수 부여 구간 마스크
    mask = (hp_series > 85) & (hp_series < 100)

    # 85% 이하와 100% 이상 처리
    normalized_hp_series[normalized_hp_series <= 85] = 0.0
    normalized_hp_series[normalized_hp_series >= 100] = 100.0

    if mask.any():
        # 시그모이드 함수 적용
        # 1. 입력값(85~100)을 특정 범위(예: -5 ~ 5)로 스케일링
        x = hp_series[mask]
        x_scaled = ((x - 85) / 15.0) * 10.0 - 5.0

        # 2. 시그모이드 함수 계산
        sigmoid_vals = 1 / (1 + np.exp(-x_scaled))

        # 3. 시그모이드 결과(sigmoid(-5) ~ sigmoid(5))를 0~100 범위로 스케일링
        s_min = 1 / (1 + np.exp(5.0))
        s_max = 1 / (1 + np.exp(-5.0))

        scaled_scores = (sigmoid_vals - s_min) / (s_max - s_min) * 100
        normalized_hp_series[mask] = scaled_scores

    return normalized_hp_series

def run_leader_analysis(stock_df, analysis_days, target_alpha, avg_trade_val_threshold, status_text, progress_bar):
    """주어진 종목들에 대해 주도주 분석을 수행하고 결과를 반환합니다."""
    kospi_code = "^KS11" # 주도주 분석은 KOSPI 지수를 벤치마크로 고정

    try:
        # 1. 종목 리스트 확보
        tickers = stock_df['Code'].tolist()
        if not tickers:
            st.warning("분석할 대상 종목이 없습니다.")
            return None
            
        name_map = stock_df.set_index('Code')['Name'].to_dict()
        marcap_map = stock_df.set_index('Code')['Marcap'].to_dict()
        market_map = stock_df.set_index('Code')['Market'].to_dict()

        # 야후 파이낸스 티커 형식으로 변환 (+ KOSPI 지수 추가)
        yahoo_tickers = [to_yahoo_ticker(code, market_map.get(code)) for code in tickers]
        all_yahoo_tickers = [kospi_code] + yahoo_tickers

        progress_bar.progress(10, text="Step 1. 분석 대상 확정")

        # ==========================================
        # 2. 데이터 다운로드 (야후 파이낸스 Rate Limit 방지 스텔스 모드)
        # ==========================================
        status_text.text(f"Step 2: {len(all_yahoo_tickers)}개 종목 데이터 다운로드 중... (서버 과부하 방지 분할 다운로드)")
        today = datetime.date.today()
        start_period = today - datetime.timedelta(days=380) # 250일 최고가 계산을 위해 기간 확장
        
        chunk_size = 100 # 한 번에 100개씩만 요청 (안전 제일)
        data_frames = []
        total_tickers = len(all_yahoo_tickers)
        
        for i in range(0, total_tickers, chunk_size):
            chunk = all_yahoo_tickers[i : i + chunk_size]
            
            try:
                # 100개씩 다운로드
                chunk_data = yf.download(chunk, start=start_period, end=today + datetime.timedelta(days=1), progress=False, ignore_tz=True)
                
                # 성공적으로 가져온 경우 보관함(리스트)에 추가
                if not chunk_data.empty:
                    data_frames.append(chunk_data)
                
                # 야후 서버가 눈치채지 못하게 0.5초 숨 고르기
                time.sleep(0.5)
                
            except Exception as e:
                # 특정 조각에서 에러가 나도 앱이 죽지 않고 다음 조각으로 넘어가도록 방어
                print(f"다운로드 일부 실패 (무시하고 진행): {e}")
                continue
                
            # 진행률 바 부드럽게 업데이트 (10% ~ 50% 구간)
            current_progress = 10 + int(40 * (min(i + chunk_size, total_tickers) / total_tickers))
            progress_bar.progress(current_progress, text=f"Step 2. 데이터 다운로드 중... ({min(i + chunk_size, total_tickers)}/{total_tickers})")

        # 쪼개서 받은 데이터 조각들을 퍼즐 맞추듯 하나로 합치기 (옆으로 이어 붙임)
        if data_frames:
            data = pd.concat(data_frames, axis=1)
            # 합치는 과정에서 혹시 중복된 컬럼이 생겼다면 제거
            data = data.loc[:, ~data.columns.duplicated()]
        else:
            st.error("야후 서버에서 데이터를 전혀 가져오지 못했습니다. IP 차단이 풀릴 때까지 잠시(약 10~30분) 기다려주세요.")
            return None
            
        progress_bar.progress(50, text="Step 2. 데이터 다운로드 완료")

# 3. 데이터 전처리 및 계산
        status_text.text("Step 3: 주도주 점수 및 기관 매집 흔적 계산 중...")
        open_df, close_df, volume_df = data['Open'], data['Close'], data['Volume']

        # KOSPI 데이터 분리
        kospi_open = open_df.pop(kospi_code) if kospi_code in open_df.columns else pd.Series()
        kospi_close = close_df.pop(kospi_code) if kospi_code in close_df.columns else pd.Series()
        if kospi_code in volume_df.columns: volume_df.pop(kospi_code)

        if kospi_close.empty:
            st.error("KOSPI 벤치마크 데이터를 가져올 수 없습니다. 분석을 중단합니다.")
            return None

        # [지표 1] 신고가 근접도 계산 (전체 기간)
        high_250d = close_df.tail(250).max()
        current_prices_for_proximity = close_df.iloc[-1]
        raw_high_proximity = (current_prices_for_proximity / high_250d) * 100
        high_proximity_normalized = normalize_high_proximity_score(raw_high_proximity)

        required_data_points = analysis_days + 1
        valid_tickers_mask = open_df.count() >= required_data_points
        open_df, close_df, volume_df = open_df.loc[:, valid_tickers_mask], close_df.loc[:, valid_tickers_mask], volume_df.loc[:, valid_tickers_mask]

        # 🌟 [새로운 지표] 기관 매집 흔적 (Volume Spike) 계산
        # 최근 20일 이동평균 거래량 계산
        vol_ma20 = volume_df.rolling(window=20, min_periods=1).mean()
        # 거래량이 20일 평균의 2배(200%) 이상 터진 날을 True로 체크
        volume_spike_mask = volume_df > (vol_ma20 * 2.0)

        # 분석 기간 자르기
        open_df, close_df, volume_df = open_df.tail(required_data_points), close_df.tail(required_data_points), volume_df.tail(required_data_points)
        kospi_open, kospi_close = kospi_open.tail(required_data_points), kospi_close.tail(required_data_points)
        volume_spike_mask = volume_spike_mask.tail(required_data_points)

        stock_base_prices, kospi_base_price = open_df.iloc[0], kospi_open.iloc[0]

        recent_period_start_index = round(required_data_points * (10 / 31))
        analysis_period_df = close_df.iloc[recent_period_start_index:]
        analysis_kospi_close = kospi_close.iloc[recent_period_start_index:]
        analysis_volume_df = volume_df.iloc[recent_period_start_index:]
        analysis_spike_mask = volume_spike_mask.iloc[recent_period_start_index:]

        # 일별 Alpha 계산
        stock_roi_daily = (analysis_period_df - stock_base_prices) / stock_base_prices * 100
        kospi_roi_daily = (analysis_kospi_close - kospi_base_price) / kospi_base_price * 100
        daily_alphas = stock_roi_daily.subtract(kospi_roi_daily, axis=0)

        # [지표 2] Alpha Persistence 계산
        stock_daily_returns = analysis_period_df.pct_change()
        kospi_daily_returns = analysis_kospi_close.pct_change()
        daily_alpha_for_persistence = stock_daily_returns.subtract(kospi_daily_returns, axis=0)
        raw_alpha_persistence = (daily_alpha_for_persistence > 0).sum(axis=0) / len(daily_alpha_for_persistence.dropna()) * 100

        min_ap, max_ap = raw_alpha_persistence.min(), raw_alpha_persistence.max()
        alpha_persistence_normalized = ((raw_alpha_persistence - min_ap) / (max_ap - min_ap)) * 100 if max_ap > min_ap else pd.Series(0.0, index=raw_alpha_persistence.index)

        # 🌟 [새로운 지표 점수화] Volume Spike 점수 (0~100점)
        # 분석 기간 중 거래량이 폭발한 날의 횟수를 구함
        spike_counts = analysis_spike_mask.sum(axis=0)
        # 매집이 가장 많았던 종목을 100점으로 기준 잡아 상대평가 (최소 1번 이상 터진 것들만)
        max_spikes = spike_counts.max()
        if max_spikes > 0:
            volume_spike_normalized = (spike_counts / max_spikes) * 100
        else:
            volume_spike_normalized = pd.Series(0.0, index=spike_counts.index)

        # 하이브리드 가중치 및 최종 점수(Alpha) 계산
        trade_val_df = analysis_period_df * analysis_volume_df
        avg_trade_val_per_stock = trade_val_df.mean(axis=0)
        avg_trade_val_per_stock[avg_trade_val_per_stock == 0] = 1
        growth_rate_weights = trade_val_df.div(avg_trade_val_per_stock, axis=1)
        log_scale_weights = np.log1p(trade_val_df)
        hybrid_raw_weights = (growth_rate_weights * log_scale_weights).clip(lower=0)
        total_hybrid_weights_per_stock = hybrid_raw_weights.sum(axis=0)
        total_hybrid_weights_per_stock[total_hybrid_weights_per_stock == 0] = 1
        daily_weights = hybrid_raw_weights.div(total_hybrid_weights_per_stock, axis=1)

        final_scores = (daily_alphas * daily_weights).sum(axis=0)
        progress_bar.progress(80, text="Step 3. 점수 계산 완료")

        # ---------------------------------------------------------
        # Step 4: 결과 필터링 및 포맷팅 부분 수정
        # ---------------------------------------------------------
        status_text.text("Step 4: 결과 정리 및 거래대금 필터링 중...")
        
        # 평균 거래대금 계산 (원 단위)
        avg_trade_vals_raw = (close_df * volume_df).mean(axis=0)
        threshold_won = avg_trade_val_threshold * 100_000_000 # 억 단위를 원 단위로 변환
        
        # 1. Alpha 점수 조건 + 2. 평균거래대금 조건 동시 만족 필터링
        trade_val_mask = avg_trade_vals_raw >= threshold_won
        high_score_series = final_scores[(final_scores >= target_alpha) & (trade_val_mask.reindex(final_scores.index, fill_value=False))]

        leader_stocks = []
        if not high_score_series.empty:
            # Alpha 점수 랭크 기반 정규화 (아웃라이어 왜곡 방지)
            normalized_scores = high_score_series.rank(pct=True) * 100

            ap_scores = alpha_persistence_normalized.reindex(high_score_series.index).fillna(0)
            hp_scores = high_proximity_normalized.reindex(high_score_series.index).fillna(0)
            vs_scores = volume_spike_normalized.reindex(high_score_series.index).fillna(0)
            
            # 🏆 새로운 주도주 통합 점수 가중치 (Alpha 35%, 매물대 25%, 승률 25%, 매집 15%)
            final_leader_score = (0.35 * normalized_scores) + (0.25 * ap_scores) + (0.25 * hp_scores) + (0.15 * vs_scores)

            yahoo_to_code_map = {v: k for k, v in {code: to_yahoo_ticker(code, market) for code, market in market_map.items()}.items()}
            filtered_codes = [yahoo_to_code_map.get(yc) for yc in high_score_series.index if yahoo_to_code_map.get(yc)]
            
            stock_total_roi = (close_df.iloc[-1] - stock_base_prices) / stock_base_prices * 100
            benchmark_total_roi = (kospi_close.iloc[-1] - kospi_base_price) / kospi_base_price * 100

            result_df = pd.DataFrame({'코드': filtered_codes})
            result_df['종목명'] = result_df['코드'].map(name_map)
            result_df['시장'] = result_df['코드'].map(market_map)
            result_df['최종 점수'] = result_df['코드'].map(lambda c: final_leader_score.get(to_yahoo_ticker(c, market_map.get(c)))).round(2)
            result_df['가중 Alpha 점수'] = result_df['코드'].map(lambda c: normalized_scores.get(to_yahoo_ticker(c, market_map.get(c)))).round(2)
            result_df['Alpha Persistence(%)'] = result_df['코드'].map(lambda c: alpha_persistence_normalized.get(to_yahoo_ticker(c, market_map.get(c)))).round(2)
            result_df['신고가 근접도(%)'] = result_df['코드'].map(lambda c: high_proximity_normalized.get(to_yahoo_ticker(c, market_map.get(c)))).round(2)
            result_df['매집 흔적(점수)'] = result_df['코드'].map(lambda c: volume_spike_normalized.get(to_yahoo_ticker(c, market_map.get(c)))).round(2)
            result_df['수익률(%)'] = result_df['코드'].map(lambda c: stock_total_roi.get(to_yahoo_ticker(c, market_map.get(c)))).round(2)
            result_df['시가총액'] = (result_df['코드'].map(marcap_map).fillna(0) / 100000000).astype(int)

            avg_trade_vals = (close_df * volume_df).mean() / 100000000
            result_df['평균거래대금'] = result_df['코드'].map(lambda c: avg_trade_vals.get(to_yahoo_ticker(c, market_map.get(c)))).fillna(0).astype(int)

            leader_stocks = result_df.to_dict('records')

        progress_bar.progress(100, text="Step 4. 분석 완료!")

        # 5. 결과 반환
        return {
            'leader_stocks': leader_stocks,
            'open_df': open_df,
            'close_df': close_df,
            'kospi_open': kospi_open,
            'kospi_close': kospi_close,
            'benchmark_roi': benchmark_total_roi,
            'full_data': data,
            's_date': open_df.index[0].date(),
            'e_date': open_df.index[-1].date(),
            'analysis_days': analysis_days
        }

    except Exception as e:
        st.error(f"주도주 분석 중 오류가 발생했습니다: {e}")
        return None

# ---------------------------------------------------------
# 캐싱 함수: 데이터 다운로드는 반복 실행 시 캐시를 사용하여 속도 향상
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_listing(market):
    """
    시장(KOSPI, KOSDAQ)의 종목 리스트를 3중 우회 로직으로 가져옵니다.
    Plan A(FDR) -> Plan B(pykrx) -> Plan C(KIND)
    """
    csv_file = f'{market.lower()}_tickers.csv'
    
    # [공통] 유효한 캐시 파일이 있으면 바로 사용
    if os.path.exists(csv_file) and (time.time() - os.path.getmtime(csv_file)) < 86400:
        return pd.read_csv(csv_file, dtype={'Code': str})

    # ==========================================
    # [플랜 A] FinanceDataReader 시도
    # ==========================================
    try:
        df = fdr.StockListing(market)
        if not df.empty and 'Code' in df.columns:
            df.to_csv(csv_file, index=False)
            return df
    except Exception as e:
        st.warning(f"⚠️ {market} 1차 수집(FDR) 실패. 플랜 B(pykrx)로 우회합니다...")
        pass # 에러 무시하고 다음 플랜으로

    # ==========================================
    # [플랜 B] pykrx 우회 시도
    # ==========================================
    try:
        for i in range(5):
            target_date = (datetime.date.today() - datetime.timedelta(days=i)).strftime("%Y%m%d")
            cap_df = stock.get_market_cap(target_date, market=market)
            if not cap_df.empty:
                break
        
        if not cap_df.empty:
            cap_df = cap_df.reset_index()
            cap_df.rename(columns={'티커': 'Code', '시가총액': 'Marcap'}, inplace=True)
            cap_df['Name'] = cap_df['Code'].apply(lambda x: stock.get_market_ticker_name(x))
            cap_df['Market'] = market
            
            result_df = cap_df[['Code', 'Name', 'Market', 'Marcap']]
            result_df.to_csv(csv_file, index=False)
            return result_df
        else:
            raise ValueError("pykrx 응답이 비어있습니다.")
            
    except Exception as e2:
        st.warning(f"⚠️ {market} 2차 수집(pykrx) 실패. 최후의 보루 플랜 C(KIND)로 우회합니다...")
        pass # 에러 무시하고 마지막 플랜으로

    # ==========================================
    # [플랜 C] KIND 전자공시시스템 직접 호출 (최후의 보루)
    # ==========================================
    try:
        mkt_type = 'stockMkt' if market == 'KOSPI' else 'kosdaqMkt'
        url = f"http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13&marketType={mkt_type}"
        
        # 🛡️ 신분 위장 헤더
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        
        df = pd.read_html(io.StringIO(response.text), header=0)[0]
        
        df = df[['회사명', '종목코드']]
        df = df.rename(columns={'회사명': 'Name', '종목코드': 'Code'})
        df['Code'] = df['Code'].astype(str).str.zfill(6)
        df['Market'] = market
        df['Marcap'] = 999900000000 
        
        df.to_csv(csv_file, index=False)
        st.success(f"✅ 플랜 C(KIND) 위장 수집 성공!")
        return df

    except Exception as e3:
        if os.path.exists(csv_file):
            st.error(f"모든 서버 접속 실패. 기존 캐시된 {market} 리스트를 사용합니다.")
            return pd.read_csv(csv_file, dtype={'Code': str})
            
        st.error(f"데이터 수집에 완전히 실패했습니다. (원인: {e3})")
        return pd.DataFrame(columns=['Code', 'Name', 'Market', 'Marcap'])

@st.cache_data(ttl=86400, show_spinner=False)
def get_admin_stocks():
    """관리종목, 거래정지 종목 등 투자 유의 종목을 안전하게 가져옵니다."""
    csv_file = 'krx_admin_tickers.csv'
    try:
        if os.path.exists(csv_file) and (time.time() - os.path.getmtime(csv_file)) < 86400:
            admin_df = pd.read_csv(csv_file, dtype=str)
            col_name = 'Symbol' if 'Symbol' in admin_df.columns else 'Code'
            return admin_df[col_name].tolist()

        # 오타 수정됨!
        admin_df = fdr.StockListing('KRX-ADMIN')
        admin_df.to_csv(csv_file, index=False)
        col_name = 'Symbol' if 'Symbol' in admin_df.columns else 'Code'
        return admin_df[col_name].tolist()

    except Exception as e:
        if os.path.exists(csv_file):
            admin_df = pd.read_csv(csv_file, dtype=str)
            col_name = 'Symbol' if 'Symbol' in admin_df.columns else 'Code'
            return admin_df[col_name].tolist()
        
        # 최악의 경우 빈 리스트를 반환하여 분석이 멈추지 않게 방어
        st.warning("⚠️ 관리종목 데이터를 가져오지 못해 필터링 없이 진행합니다.")
        return []

st.title("📈 KOSPI Alpha Tracker")
st.markdown("KOSPI 지수 대비 초과 수익(Alpha)을 달성한 종목을 발굴합니다.")

# ---------------------------------------------------------
# 관심 종목 관리 기능 (로컬 저장 및 로딩)
# ---------------------------------------------------------
WATCHLIST_FILE = 'watchlist.csv'

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            return pd.read_csv(WATCHLIST_FILE, dtype={'Code': str}).to_dict('records')
        except: return []
    return []

def save_watchlist(watchlist_data):
    if watchlist_data:
        pd.DataFrame(watchlist_data).to_csv(WATCHLIST_FILE, index=False)
    elif os.path.exists(WATCHLIST_FILE):
        os.remove(WATCHLIST_FILE)

if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = load_watchlist()

with st.expander("⭐ 나의 관심 종목", expanded=False):
    # --- 관심 종목 추가 기능 ---
    st.markdown("##### 종목 직접 추가")
    
    # 종목 리스트 로딩 (캐시된 get_listing 함수 사용)
    @st.cache_data
    def get_all_stocks_for_selection():
        kospi_df = get_listing('KOSPI')
        kospi_df['Market'] = 'KOSPI'
        kosdaq_df = get_listing('KOSDAQ')
        kosdaq_df['Market'] = 'KOSDAQ'
        all_stocks = pd.concat([kospi_df, kosdaq_df], ignore_index=True)
        all_stocks['Label'] = all_stocks['Name'] + " (" + all_stocks['Code'] + ")"
        return all_stocks

    all_stocks_df = get_all_stocks_for_selection()
    all_stock_options = all_stocks_df['Label'].tolist()

    col1_add, col2_add = st.columns([3, 1])
    selected_stock_label = col1_add.selectbox("추가할 종목 검색", all_stock_options, index=None, placeholder="종목명 또는 코드로 검색...", label_visibility="collapsed")

    if col2_add.button("➕ 종목 추가", width='stretch'):
        if selected_stock_label:
            selected_stock_row = all_stocks_df[all_stocks_df['Label'] == selected_stock_label].iloc[0]
            new_item = {'Code': selected_stock_row['Code'], 'Name': selected_stock_row['Name'], 'Market': selected_stock_row['Market']}
            if not any(item['Code'] == new_item['Code'] for item in st.session_state['watchlist']):
                st.session_state['watchlist'].append(new_item)
                save_watchlist(st.session_state['watchlist'])
                st.success(f"'{new_item['Name']}' 종목이 관심 종목에 등록되었습니다.")
                # 관심종목 추가 시, 해당 종목을 포함하여 재분석
                st.session_state.analysis_needed = 'watchlist'
                st.rerun()
            else:
                st.warning("이미 관심 종목에 등록되어 있습니다.")

    st.divider()
    if st.session_state['watchlist']:
        col_watch_table, col_watch_chart = st.columns([2, 3])

        with col_watch_table:
            watchlist_df = pd.DataFrame(st.session_state['watchlist'])
            display_df = watchlist_df.copy() # 기본 표시용 데이터프레임

            # 주도주 분석 결과가 있으면, 테이블 정보를 풍부하게 만듭니다.
            if 'leader_analysis_result' in st.session_state and st.session_state['leader_analysis_result']['leader_stocks']:
                res = st.session_state['leader_analysis_result']
                leader_df = pd.DataFrame(res['leader_stocks'])
                
                # 관심 종목(왼쪽) 기준으로 주도주 정보(오른쪽)를 합칩니다.
                merged_df = pd.merge(watchlist_df, leader_df, how='left', left_on='Code', right_on='코드')

                # 주도주 테이블과 동일한 컬럼 순서 및 이름으로 재구성
                display_cols = ['Name', 'Code', 'Market', '최종 점수', '가중 Alpha 점수', 'Alpha Persistence(%)', '신고가 근접도(High Proximity)(%)', '수익률(%)', '시가총액', '평균거래대금']
                display_df = merged_df.reindex(columns=display_cols)
                display_df.rename(columns={'Name': '종목명', 'Code': '코드', 'Market': '시장'}, inplace=True)

                # 10개까지만 보여주기
                display_df = display_df.head(10)
                
                styler = display_df.style.format({
                    "최종 점수": "{:.2f}", "가중 Alpha 점수": "{:.2f}",
                    "Alpha Persistence(%)": "{:.2f}%", "신고가 근접도(High Proximity)(%)": "{:.2f}%", "수익률(%)": "{:.2f}%",
                    "시가총액": "{:,.0f}억원", "평균거래대금": "{:,.0f}억원"
                }, na_rep='-')
                
                event_w = st.dataframe(styler, width='stretch', hide_index=True, on_select="rerun", selection_mode="single-row", key="watchlist_table")
            else:
                # 분석 전이거나 결과가 없으면 기본 정보만 표시
                display_df = display_df.head(10)
                event_w = st.dataframe(display_df, width='stretch', hide_index=True, on_select="rerun", selection_mode="single-row", key="watchlist_table")
            st.markdown("##### 등록된 종목 삭제")
            remove_options = [f"{item['Name']} ({item['Code']})" for item in st.session_state['watchlist']]
            col1_rem, col2_rem = st.columns([3, 1])
            sel_to_remove = col1_rem.selectbox("삭제할 종목 선택", remove_options, label_visibility="collapsed", key="watchlist_remove_sel")
            if col2_rem.button("❌ 관심 종목 삭제", width='stretch'):
                code_to_rem = sel_to_remove.split('(')[1].split(')')[0]
                st.session_state['watchlist'] = [i for i in st.session_state['watchlist'] if i['Code'] != code_to_rem]
                save_watchlist(st.session_state['watchlist'])
                st.rerun()

        with col_watch_chart:
            # 관심 종목 차트 출력 로직
            if event_w.selection.rows:
                selected_idx = event_w.selection.rows[0]
                row = display_df.iloc[selected_idx]
                sel_code, sel_market, sel_name = (row['코드'], row['시장'], row['종목명']) if '코드' in row else (row['Code'], row['Market'], row['Name'])
                sel_yahoo_code = to_yahoo_ticker(sel_code, sel_market)
                
                ticker_data = None
                kospi_series = None

                # 1. 분석 결과에 데이터가 있는지 확인
                if 'leader_analysis_result' in st.session_state:
                    res = st.session_state['leader_analysis_result']
                    if sel_yahoo_code in res['full_data']['Close'].columns:
                        full_data = res['full_data']
                        ticker_data = pd.DataFrame({
                            'Open': full_data['Open'][sel_yahoo_code],
                            'High': full_data['High'][sel_yahoo_code],
                            'Low': full_data['Low'][sel_yahoo_code],
                            'Close': full_data['Close'][sel_yahoo_code],
                            'Volume': full_data['Volume'][sel_yahoo_code]
                        }).tail(60).dropna()
                        
                        if not res['kospi_close'].empty:
                            kospi_series = res['kospi_close'].loc[ticker_data.index.intersection(res['kospi_close'].index)]

                # 2. 데이터가 없으면 새로 다운로드
                if ticker_data is None or ticker_data.empty:
                    with st.spinner(f"'{sel_name}' 종목의 최신 데이터를 가져오는 중..."):
                        today = datetime.date.today()
                        start_date = today - datetime.timedelta(days=90) # 60일치 데이터를 위해 넉넉하게 90일 다운
                        
                        try:
                            chart_data = yf.download([sel_yahoo_code, "^KS11"], start=start_date, end=today + datetime.timedelta(days=1), progress=False, ignore_tz=True)
                            
                            if not chart_data.empty and sel_yahoo_code in chart_data['Close'].columns:
                                ticker_data = pd.DataFrame({
                                    'Open': chart_data['Open'][sel_yahoo_code],
                                    'High': chart_data['High'][sel_yahoo_code],
                                    'Low': chart_data['Low'][sel_yahoo_code],
                                    'Close': chart_data['Close'][sel_yahoo_code],
                                    'Volume': chart_data['Volume'][sel_yahoo_code]
                                }).tail(60).dropna()
                                
                                if not ticker_data.empty and "^KS11" in chart_data['Close'].columns:
                                    kospi_data_full = chart_data['Close']['^KS11'].dropna()
                                    kospi_series = kospi_data_full.loc[ticker_data.index.intersection(kospi_data_full.index)]
                                else:
                                    kospi_series = pd.Series()
                            else:
                                st.warning(f"'{sel_name}' 종목의 차트 데이터를 가져올 수 없습니다.")
                        except Exception as e:
                            st.error(f"데이터 다운로드 중 오류 발생: {e}")
                            ticker_data = None

                # 3. 차트 그리기
                if ticker_data is not None and not ticker_data.empty:
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                                           subplot_titles=(f"{sel_name} ({sel_code}) 차트", '거래량'), row_heights=[0.7, 0.3],
                                           specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
                        
                        fig.add_trace(go.Candlestick(x=ticker_data.index, open=ticker_data['Open'], high=ticker_data['High'],
                                                   low=ticker_data['Low'], close=ticker_data['Close'], name='Price'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Close'].rolling(5).mean(), name='MA5', line=dict(width=1)), row=1, col=1)
                        if kospi_series is not None and not kospi_series.empty:
                            fig.add_trace(go.Scatter(x=kospi_series.index, y=kospi_series, name='KOSPI', line=dict(color='purple', width=1, dash='dot')), row=1, col=1, secondary_y=True)
                        fig.add_trace(go.Bar(x=ticker_data.index, y=ticker_data['Volume'], name='Volume', marker_color='gray'), row=2, col=1)
                        fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(t=30, b=10))

                        fig.update_xaxes(type='category') # 휴장일 제거

                        tickvals = ticker_data.index[::5]
                        fig.update_xaxes(tickmode='array', tickvals=tickvals, ticktext=[d.strftime('%Y-%m-%d') for d in tickvals])

                        if kospi_series is not None and not kospi_series.empty and not ticker_data['Close'].empty:
                            stock_start, kospi_start = ticker_data['Close'].iloc[0], kospi_series.iloc[0]
                            if stock_start != 0:
                                ratio = kospi_start / stock_start
                                y1_min, y1_max = ticker_data['Low'].min(), ticker_data['High'].max()
                                fig.update_layout(yaxis2=dict(range=[y1_min * ratio, y1_max * ratio], overlaying='y', side='right', title='KOSPI'))
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("테이블에서 종목을 선택하면 상세 차트가 표시됩니다.")
    else:
        st.info("등록된 관심 종목이 없습니다. 위에서 직접 추가하거나, 분석 결과에서 종목을 추가해 보세요.")

# 앱 최초 실행 시 관심종목이 있으면 자동으로 분석을 트리거합니다.
if 'analysis_run_for' not in st.session_state:
    if st.session_state.watchlist:
        st.session_state.analysis_needed = 'watchlist'


# ---------------------------------------------------------
# 사이드바: 사용자 입력
# ---------------------------------------------------------
with st.sidebar:
    st.header("설정")
    
    # 날짜 기본값 설정 (오늘 날짜 기준)
    today = datetime.date.today()
    
    market_selection = st.selectbox("시장 선택", ["KOSPI", "KOSDAQ", "전체"], index=2, help="분석할 주식 시장을 선택합니다.")

    # [성능 개선] 시가총액 필터 추가
    marcap_threshold = st.slider("최소 시가총액 (억원)", 0, 10000, 4000, 100, help="시가총액이 이 값 미만인 종목은 분석에서 제외하여 속도를 향상시킵니다.")
    
    # [설정 추가] 평균거래대금 필터
    avg_trade_val_threshold = st.slider(
        "최소 평균거래대금 (억원)", 
        0, 1000, 100, 10, 
        help="최근 분석 기간 동안의 일평균 거래대금이 이 값 미만인 종목은 제외합니다."
    )
    analysis_days = st.slider("분석 기간 (거래일)", min_value=10, max_value=120, value=30, step=1, help="주도주 분석에 사용할 거래일 수를 설정합니다.")

    target_alpha = st.number_input("목표 Alpha (%)", value=10.0, step=0.5)
    
    find_leader_btn = st.button("주도주 찾기")
    if find_leader_btn:
        st.session_state.analysis_needed = 'full'
        st.rerun()

# ---------------------------------------------------------
# 주도주 분석 실행 로직
# ---------------------------------------------------------
analysis_trigger = st.session_state.pop('analysis_needed', None)
if analysis_trigger:
    status_text = st.empty()
    progress_bar = st.progress(0)
    stock_df = pd.DataFrame()

    # 1. 분석 대상 종목 DataFrame 생성
    if analysis_trigger == 'full':
        status_text.text("Step 1: 분석 대상 종목 리스트 확보 중...")
        if market_selection == 'KOSPI':
            stock_df = get_listing('KOSPI'); stock_df['Market'] = 'KOSPI'
        elif market_selection == 'KOSDAQ':
            stock_df = get_listing('KOSDAQ'); stock_df['Market'] = 'KOSDAQ'
        else: # '전체'
            kospi_df = get_listing('KOSPI'); kospi_df['Market'] = 'KOSPI'
            kosdaq_df = get_listing('KOSDAQ'); kosdaq_df['Market'] = 'KOSDAQ'
            stock_df = pd.concat([kospi_df, kosdaq_df], ignore_index=True)

        stock_df = stock_df[stock_df['Marcap'] >= (marcap_threshold * 100000000)]
        stock_df = stock_df[~stock_df['Code'].isin(get_admin_stocks())]
    
    elif analysis_trigger == 'watchlist' and st.session_state.watchlist:
        status_text.text("Step 1: 관심 종목 리스트 확보 중...")
        watchlist_codes = [item['Code'] for item in st.session_state.watchlist]
        kospi_df = get_listing('KOSPI'); kospi_df['Market'] = 'KOSPI'
        kosdaq_df = get_listing('KOSDAQ'); kosdaq_df['Market'] = 'KOSDAQ'
        all_stocks_df = pd.concat([kospi_df, kosdaq_df], ignore_index=True)
        stock_df = all_stocks_df[all_stocks_df['Code'].isin(watchlist_codes)].copy()

# 2. 분석 실행
    if not stock_df.empty:
        result = run_leader_analysis(stock_df, analysis_days, target_alpha, avg_trade_val_threshold, status_text, progress_bar)
        if result:
            st.session_state['leader_analysis_result'] = result
    else:
        if analysis_trigger == 'full': st.warning("선택된 조건에 맞는 분석 대상 종목이 없습니다.")
        if 'leader_analysis_result' in st.session_state: del st.session_state['leader_analysis_result']

    # 핵심: 분석이 성공하든 실패하든 '실행했다'는 기록을 무조건 남겨서 무한 루프를 끊습니다.
    st.session_state.analysis_run_for = analysis_trigger
    
    st.rerun()

# ---------------------------------------------------------
# 주도주 분석 결과 출력
# ---------------------------------------------------------
if 'leader_analysis_result' in st.session_state:
    res = st.session_state['leader_analysis_result']
    leader_stocks = res['leader_stocks']
    analysis_days = res['analysis_days']
    open_df = res['open_df']
    close_df = res['close_df']
    kospi_open = res['kospi_open']
    kospi_close = res['kospi_close']
    benchmark_roi = res['benchmark_roi']
    s_date = res['s_date']
    e_date = res['e_date']

    st.info(f"📊 {analysis_days}거래일 기준 시장 수익률 (KOSPI): **{benchmark_roi:.2f}%**")

    if leader_stocks:
        result_df = pd.DataFrame(leader_stocks)
        result_df = result_df.sort_values(by='최종 점수', ascending=False).reset_index(drop=True)

        total_found = len(result_df)

        st.success(f"🚀 총 {total_found}개의 주도주 후보 종목 발견!")

        # 화면에 보여줄 컬럼 리스트에 '매집 흔적(점수)' 추가
        display_cols = [
            '종목명', '코드', '시장', '최종 점수', '가중 Alpha 점수', 'Alpha Persistence(%)', '신고가 근접도(%)', '매집 흔적(점수)', '수익률(%)',
            '시가총액', '평균거래대금'
        ]
        result_df_display = result_df.reindex(columns=display_cols).fillna(0)
        
        styler = result_df_display.style.format({
            "최종 점수": "{:.2f}", "가중 Alpha 점수": "{:.2f}", "Alpha Persistence(%)": "{:.2f}%", 
            "신고가 근접도(%)": "{:.2f}%", "매집 흔적(점수)": "{:.0f}점", "수익률(%)": "{:.2f}%",
            "시가총액": "{:,.0f}억원", "평균거래대금": "{:,.0f}억원"
        })

        col_leader_table, col_leader_chart = st.columns([3, 2])

        with col_leader_table:
            event = st.dataframe(
                styler, width='stretch', hide_index=True,
                on_select="rerun", selection_mode="single-row", key="leader_stock_table"
            )
            
            # 관심 종목 등록 버튼
            if event.selection.rows:
                selected_idx = event.selection.rows[0]

            # 엑셀 다운로드
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                result_df.to_excel(writer, index=False, sheet_name='LeaderStocks')
            st.download_button(
                label="📥 엑셀 파일 다운로드", data=buffer,
                file_name=f"leader_stocks_{e_date}.xlsx", mime="application/vnd.ms-excel"
            )

        with col_leader_chart:
            st.subheader(f"📈 {analysis_days}거래일 수익률 상세 비교")
            
            option_list = [f"{row['종목명']} ({row['코드']})" for _, row in result_df.iterrows()]
            code_map = {label: code for label, code in zip(option_list, result_df['코드'])}
            market_map_from_df = result_df.set_index('코드')['시장'].to_dict()
            
            if 'leader_last_selection_rows' not in st.session_state: st.session_state.leader_last_selection_rows = []
            current_selection_rows = event.selection.rows
            selectbox_key = "leader_chart_ticker_select"
            
            if current_selection_rows != st.session_state.leader_last_selection_rows:
                if current_selection_rows:
                    idx = current_selection_rows[0]
                    if idx < len(option_list): st.session_state[selectbox_key] = option_list[idx]
                st.session_state.leader_last_selection_rows = current_selection_rows

            selected_label = st.selectbox("차트로 확인할 종목을 선택하세요:", option_list, key=selectbox_key)
            
            if selected_label:
                sel_code = code_map[selected_label]
                sel_market = market_map_from_df[sel_code]
                sel_yahoo_code = to_yahoo_ticker(sel_code, sel_market)
                
                if sel_yahoo_code in close_df.columns:
                    # 주도주 분석용 전문 차트
                    full_data = res['full_data']
                    ticker_data = pd.DataFrame({
                        'Open': full_data['Open'][sel_yahoo_code],
                        'High': full_data['High'][sel_yahoo_code],
                        'Low': full_data['Low'][sel_yahoo_code],
                        'Close': full_data['Close'][sel_yahoo_code],
                        'Volume': full_data['Volume'][sel_yahoo_code]
                    }).tail(60).dropna()

                    # KOSPI 지수 데이터 준비
                    kospi_series = kospi_close.loc[ticker_data.index.intersection(kospi_close.index)]

                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                                       subplot_titles=(f"{selected_label} 주도주 상세 차트", '거래량'), row_heights=[0.7, 0.3],
                                       specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
                    
                    fig.add_trace(go.Candlestick(x=ticker_data.index, open=ticker_data['Open'], high=ticker_data['High'],
                                               low=ticker_data['Low'], close=ticker_data['Close'], name='Price'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Close'].rolling(5).mean(), name='MA5', line=dict(width=1)), row=1, col=1)
                    
                    # KOSPI 지수 추가 (보조 y축)
                    if not kospi_series.empty:
                        fig.add_trace(go.Scatter(x=kospi_series.index, y=kospi_series, name='KOSPI', line=dict(color='purple', width=1, dash='dot')), row=1, col=1, secondary_y=True)

                    fig.add_trace(go.Bar(x=ticker_data.index, y=ticker_data['Volume'], name='Volume', marker_color='gray'), row=2, col=1)
                    
                    fig.update_layout(xaxis_rangeslider_visible=False, height=600, margin=dict(t=30, b=10))

                    fig.update_xaxes(type='category') # 휴장일 제거

                    tickvals = ticker_data.index[::5]
                    fig.update_xaxes(tickmode='array', tickvals=tickvals, ticktext=[d.strftime('%Y-%m-%d') for d in tickvals])

                    # Y축 중심 정렬 로직
                    if not kospi_series.empty and not ticker_data['Close'].empty:
                        stock_start = ticker_data['Close'].iloc[0]
                        kospi_start = kospi_series.iloc[0]
                        
                        if stock_start != 0:
                            ratio = kospi_start / stock_start
                            y1_min, y1_max = ticker_data['Low'].min(), ticker_data['High'].max()
                            
                            fig.update_layout(
                                yaxis2=dict(
                                    range=[y1_min * ratio, y1_max * ratio],
                                    overlaying='y', side='right', title='KOSPI'
                                )
                            )

                    st.plotly_chart(fig, width='stretch')
            else:
                st.info("테이블에서 종목을 선택하면 상세 차트가 표시됩니다.")
    else:
        st.warning("조건을 만족하는 주도주 후보 종목이 없습니다.")
