import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import time
import os

# 1. 날짜 설정
start_date = "2026-01-02"
end_date = "2026-01-19"

def to_yahoo_ticker(code, market):
    """종목코드와 시장 정보를 야후 파이낸스 티커 형식으로 변환합니다."""
    return f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ"


print(f"[{start_date} ~ {end_date}] KOSPI Alpha 스캔 (고속 버전)...")

# ---------------------------------------------------------
# [Step 0] KOSPI 지수 수익률 (Baseline)
# ---------------------------------------------------------
print("0. KOSPI 지수 계산 중...")
try:
    kospi_data = pd.DataFrame()
    for i in range(3): # Rate Limit 에러 대응 재시도
        try:
            kospi_data = yf.download("^KS11", start=start_date, end=end_date, progress=False, ignore_tz=True)
            if not kospi_data.empty:
                break
        except Exception:
            time.sleep((i + 1) * 2)

    if isinstance(kospi_data.columns, pd.MultiIndex):
        k_close = kospi_data['Close']['^KS11']
    else:
        k_close = kospi_data['Close']
    
    k_close = k_close.dropna()
    kospi_roi = (k_close.iloc[-1] - k_close.iloc[0]) / k_close.iloc[0] * 100
    print(f"   -> 시장 수익률: {kospi_roi:.2f}%")
except Exception as e:
    print(f"❌ KOSPI 지수 에러: {e}")
    exit()

# ---------------------------------------------------------
# [Step 1] 종목 리스트 & 해시맵 생성 (최적화 핵심)
# ---------------------------------------------------------
print("1. KOSPI 및 KOSDAQ 종목 리스트 확보 및 최적화...")
kospi_df = fdr.StockListing('KOSPI')
kospi_df['Market'] = 'KOSPI'
kosdaq_df = fdr.StockListing('KOSDAQ')
kosdaq_df['Market'] = 'KOSDAQ'

all_stocks_df = pd.concat([kospi_df, kosdaq_df], ignore_index=True)

tickers = all_stocks_df['Code'].tolist()

# [최적화] DataFrame 검색 대신 Dictionary(Hash Map) 생성
# O(N) 검색을 O(1)로 줄여줍니다.
name_map = all_stocks_df.set_index('Code')['Name'].to_dict()
market_map = all_stocks_df.set_index('Code')['Market'].to_dict()


# ---------------------------------------------------------
# [Step 2] 데이터 다운로드
# ---------------------------------------------------------
yahoo_tickers = [to_yahoo_ticker(code, market_map.get(code)) for code in tickers]
print(f"2. {len(tickers)}개 종목 데이터 다운로드...")
# 다운로드 로그가 너무 길면 보기 싫으니 progress=False로 하고 마지막에 완료만 띄웁니다.
data = yf.download(yahoo_tickers, start=start_date, end=end_date, progress=True, ignore_tz=True)

# ---------------------------------------------------------
# [Step 3] Alpha 계산 (Progress Bar 추가)
# ---------------------------------------------------------
print("\n3. Alpha 계산 중 (초고속 벡터화 모드)...")

if isinstance(data.columns, pd.MultiIndex):
    close_df = data.get('Adj Close', data.get('Close'))
else:
    close_df = data # 여러 종목을 다운로드하면 항상 MultiIndex이지만 안전장치

# 유효한 데이터가 없는 종목 컬럼(모든 값이 NaN)은 미리 제거
close_df.dropna(axis=1, how='all', inplace=True)

# 시작가, 종료가 계산 (벡터화)
valid_close_df = close_df.dropna(how='all')

high_alpha_series = pd.Series(dtype='float64') # 빈 시리즈로 초기화

if not valid_close_df.empty:
    start_prices = valid_close_df.bfill().iloc[0]
    end_prices = valid_close_df.ffill().iloc[-1]

    # 수익률 및 Alpha 계산 (벡터화)
    stock_roi = (end_prices - start_prices) / start_prices * 100
    alpha = stock_roi - kospi_roi

    # 목표 Alpha 이상인 종목 필터링
    target_alpha = 10.0
    high_alpha_series = alpha[alpha >= target_alpha]
else:
    print("   -> 경고: 분석 기간 동안 유효한 주가 데이터가 없습니다.")

result_df = pd.DataFrame()
if not high_alpha_series.empty:
    # 결과 데이터프레임 생성
    high_alpha_yahoo_codes = high_alpha_series.index
    yahoo_to_code_map = {to_yahoo_ticker(code, market): code for code, market in market_map.items()}
    filtered_codes = [yahoo_to_code_map.get(yc) for yc in high_alpha_yahoo_codes if yahoo_to_code_map.get(yc)]

    result_df = pd.DataFrame({'코드': filtered_codes})
    result_df['종목명'] = result_df['코드'].map(name_map)
    result_df['시장'] = result_df['코드'].map(market_map)
    result_df['시작가'] = result_df['코드'].map(lambda c: start_prices.get(to_yahoo_ticker(c, market_map.get(c)))).fillna(0)
    result_df['현재가'] = result_df['코드'].map(lambda c: end_prices.get(to_yahoo_ticker(c, market_map.get(c)))).fillna(0)
    result_df['종목수익률(%)'] = result_df['코드'].map(lambda c: stock_roi.get(to_yahoo_ticker(c, market_map.get(c)))).round(2)
    result_df['상대수익률(Alpha, %)'] = result_df['코드'].map(lambda c: high_alpha_series.get(to_yahoo_ticker(c, market_map.get(c)))).round(2)
    result_df = result_df.dropna() # 혹시 모를 매핑 실패 케이스 제거

# ---------------------------------------------------------
# [Step 4] 결과 출력
# ---------------------------------------------------------
print("-" * 60)
if not result_df.empty:
    result_df = result_df.sort_values(by='상대수익률(Alpha, %)', ascending=False)
    
    count = len(result_df)
    print(f"🚀 검색 완료: Alpha {target_alpha}% 이상 종목 [{count}개] 발견!\n")
    
    # 화면 출력
    display_df = result_df.copy()
    display_df['시작가'] = display_df['시작가'].astype(int).apply(lambda x: f"{x:,}")
    display_df['현재가'] = display_df['현재가'].astype(int).apply(lambda x: f"{x:,}")
    print(display_df[['종목명', '코드', '시장', '종목수익률(%)', '상대수익률(Alpha, %)']])

    # 엑셀 저장
    file_name = "kospi_alpha_stocks_optimized.xlsx"
    result_df.to_excel(file_name, index=False)
    print(f"\n💾 저장 완료: {os.getcwd()}\\{file_name}")
else:
    print("조건을 만족하는 종목이 없습니다.")
print("-" * 60)