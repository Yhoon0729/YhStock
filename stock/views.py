# myapp/views.py
from django.conf import settings
from django.shortcuts import render
import pandas as pd
import os
import datetime
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
from pykrx import stock
from datetime import datetime, timedelta
import io
import urllib, base64

def index(request):
    # 파일 경로 설정
    top_10_csv_path = os.path.join(settings.MEDIA_ROOT, 'top_10.csv')

    # 어제 날짜 계산
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d")

    # 1) 어제 등락률 상위 10개
    if os.path.exists(top_10_csv_path):
        top_10_df = pd.read_csv(top_10_csv_path)
        # 파일에서 날짜를 확인
        last_saved_date = top_10_df['date'].iloc[0]
        if last_saved_date != yesterday:
            # 데이터 갱신
            ohlcv = stock.get_market_ohlcv(yesterday, market="KOSPI")
            top_10 = ohlcv.sort_values(by='등락률', ascending=False).head(10)
            top_10_tickers = top_10.index.tolist()
            top_10_names = [stock.get_market_ticker_name(ticker) for ticker in top_10_tickers]
            # 데이터프레임으로 저장
            top_10_df = pd.DataFrame({'date': [yesterday]*10, 'ticker': top_10_tickers, 'name': top_10_names})
            top_10_df.to_csv(top_10_csv_path, index=False)
        else:
            # 기존 데이터 사용
            top_10_names = top_10_df['name'].tolist()
    else:
        # 파일이 없을 때 데이터 생성
        ohlcv = stock.get_market_ohlcv(yesterday, market="ALL")
        top_10 = ohlcv.sort_values(by='등락률', ascending=False).head(10)
        top_10_tickers = top_10.index.tolist()
        top_10_names = [stock.get_market_ticker_name(ticker) for ticker in top_10_tickers]
        # 데이터프레임으로 저장
        top_10_df = pd.DataFrame({'date': [yesterday]*10, 'ticker': top_10_tickers, 'name': top_10_names})
        top_10_df.to_csv(top_10_csv_path, index=False)

    # 2) 코스피 데이터 로드 및 업데이트
    csv_path = os.path.join(settings.MEDIA_ROOT, 'kospi_data.csv')
    if os.path.exists(csv_path):
        kospi = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        last_saved_date = kospi.index[-1]
        next_day = last_saved_date + pd.Timedelta(days=1)
        today_date = pd.Timestamp(datetime.today().date())

        if next_day <= today_date:
            new_data = fdr.DataReader('KS11', next_day.strftime("%Y-%m-%d"))
            kospi = kospi._append(new_data)
            kospi.to_csv(csv_path)
    else:
        months_ago = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
        kospi = fdr.DataReader('KS11', months_ago)
        kospi.to_csv(csv_path)

    # Kospi graph
    plt.figure()
    kospi['Close'].plot(title='Kospi')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    kospi_graph = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    context = {
        'top_10_names' : top_10_names,
        'top_10_tickers': top_10_tickers,
        'kospi_graph': kospi_graph,
    }
    return render(request, 'index.html', context)

def list(request):
    # 날짜 설정
    today = datetime.today().strftime('%Y-%m-%d')
    yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

    # KOSPI 주식 목록 가져오기
    all_stocks = fdr.StockListing('KOSPI')

    '''
    # 모든 종목의 데이터를 가져오기
    stock_symbols = all_stocks['Code'].tolist()

    # 어제와 오늘의 데이터 불러오기
    data_today = fdr.DataReader(stock_symbols, start=today, end=today)
    data_yesterday = fdr.DataReader(stock_symbols, start=yesterday, end=yesterday)    

    # 어제와 오늘의 데이터 병합 (같은 종목끼리 병합)
    merged_data = pd.merge(data_today, data_yesterday, on='Code', suffixes=('_today', '_yesterday'))    

    # 등락률 계산
    merged_data['Change'] = merged_data['Close_today'] - merged_data['Close_yesterday']
    merged_data['ChangePercent'] = (merged_data['Change'] / merged_data['Close_yesterday']) * 100
    
    # NaN 값 제거
    merged_data = merged_data.dropna()
    '''
    
    # 상위 10개 종목 선택 (등락률 기준)
    top_10 = all_stocks.sort_values(by='ChagesRatio', ascending=False).head(10)

    # 주식 데이터 리스트로 준비
    stocks = []
    for i in range(len(top_10)):
        stock_data = {
            'code': top_10.iloc[i].Code,
            'name': top_10.iloc[i].Name,
            'close': top_10.iloc[i].Close,
            'change': top_10.iloc[i].Changes,
            'chagesRatio':top_10.iloc[i].ChagesRatio,
        }
        stocks.append(stock_data)
    # 템플릿에 데이터 전달
    return render(request, 'stock/list.html', {'stocks': stocks})

def predict(request) :
    if request.method != 'POST':
        return render(request, 'stock/predict.html')
