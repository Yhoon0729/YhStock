# myapp/views.py
from django.conf import settings
from django.http import JsonResponse
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
import yfinance as yf

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

def info(request):
    ticker = request.GET.get('ticker')
    if ticker:
        try:
            # KOSPI 주식을 위해 ".KS" 추가
            kospi_ticker = f"{ticker}.KS"
            stock = yf.Ticker(kospi_ticker)
            hist = stock.history(period="1mo")  # 1달 데이터

            if hist.empty:
                return render(request, 'stock/info.html', {'error': '데이터를 가져올 수 없습니다.'})

            # 그래프 생성
            plt.figure(figsize=(10,5))
            plt.plot(hist.index, hist['Close'])
            plt.title(f"{ticker} 주가")
            plt.xlabel('날짜')
            plt.ylabel('종가')
            plt.xticks(rotation=45)
            plt.tight_layout()

            # 그래프를 이미지로 변환
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            graphic = base64.b64encode(image_png)
            graphic = graphic.decode('utf-8')

            plt.close()  # 메모리 누수 방지를 위해 plt 객체 닫기

            # 주식 정보 가져오기
            info = stock.info
            company_name = info.get('longName', 'N/A')
            current_price = info.get('currentPrice', 'N/A')
            previous_close = info.get('previousClose', 'N/A')

            return render(request, 'stock/info.html', {
                'graphic': graphic,
                'ticker': ticker,
                'company_name': company_name,
                'current_price': current_price,
                'previous_close': previous_close
            })

        except Exception as e:
            return render(request, 'stock/info.html', {'error': f'오류 발생: {str(e)}'})
    else:
        return render(request, 'stock/info.html')


def search_stocks(request):
    query = request.GET.get('query', '').strip()
    if query:
        try:
            # KOSPI 주식 목록을 가져옵니다
            all_stocks = fdr.StockListing('KOSPI')

            # 주식 코드가 query로 시작하는 항목을 필터링합니다
            results = all_stocks[all_stocks['Code'].str.startswith(query)]

            # 결과를 리스트로 변환합니다 (최대 5개)
            stocks = [{'code': row['Code'], 'name': row['Name']}
                      for _, row in results.head(5).iterrows()]

            return JsonResponse({'stocks': stocks})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'stocks': []})
