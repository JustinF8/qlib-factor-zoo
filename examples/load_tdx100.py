# -*- coding: utf-8 -*-
"""Load 100 stocks TQ data and save"""
import sys, os, time, threading, pandas as pd

sys.path.insert(0, r'd:/zd_zyb(x64 26011715)GA/PYPlugins/user')
from tqcenter import tq
tq.initialize(r'd:/zd_zyb(x64 26011715)GA/PYPlugins/user')

# 沪深300前100只
codes = ['000001','000002','000063','000066','000069','000100','000157','000166','000301','000333',
         '000338','000408','000425','000538','000568','000596','000617','000625','000630','000651',
         '000661','000708','000725','000768','000776','000786','000792','000800','000807','000858',
         '000876','000895','000938','000963','000977','000983','000999','001289','001965','001979',
         '002001','002027','002049','002050','002074','002129','002142','002179','002180','002230',
         '002236','002241','002252','002271','002304','002311','002352','002371','002410','002415',
         '002459','002460','002463','002466','002475','002493','002594','002601','002648','002709',
         '002714','002736','002812','002841','002916','002920','002938','003816','300014','300015',
         '300033','300059','300124','300274','300308','300316','300347','300408','300413','300433',
         '300442','300450','300498','300502','300529','300628','300661','300750','300751','300759']

all_data = []
ok = 0
for i, code in enumerate(codes):
    if code.startswith(('6','9')): ctdx = f'{code}.SH'
    elif code.startswith(('0','3','2')): ctdx = f'{code}.SZ'
    elif code.startswith('8'): ctdx = f'{code}.SZ' if code.startswith('83') else f'{code}.SH'
    else: ctdx = f'{code}.SH'

    print(f'{i+1}/{len(codes)} {code}', end=' ', flush=True)
    result = {}
    def fetch():
        try:
            result['data'] = tq.get_market_data(
                field_list=['Open','High','Low','Close','Volume'],
                stock_list=[ctdx], period='1d',
                start_time='20230101', end_time='20260531')
        except Exception as e:
            result['error'] = str(e)
    t = threading.Thread(target=fetch, daemon=True)
    t.start()
    t.join(timeout=8)
    if t.is_alive():
        print('TIMEOUT')
        continue
    if 'error' in result:
        print('ERR')
        continue
    data = result.get('data')
    if data is None:
        print('NONE')
        continue
    df = pd.DataFrame()
    for tf, of in [('Open','open'),('High','high'),('Low','low'),('Close','close'),('Volume','volume')]:
        if tf in data:
            fd = data[tf]
            if isinstance(fd, pd.DataFrame):
                df[of] = fd[ctdx] if ctdx in fd.columns else fd.iloc[:,0]
    if not df.empty:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().reset_index().rename(columns={'index':'date'})
        df['code'] = code
        all_data.append(df[['date','code','open','close','high','low','volume']])
        ok += 1
        print(f'OK {len(df)}r')
    else:
        print('EMPTY')

tq.close()
full = pd.concat(all_data, ignore_index=True)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tdx_data_100.parquet')
full.to_parquet(out, index=False)
print(f'\nSaved {len(full)} rows, {ok} stocks to {out}')
print(full.groupby('code').size().describe())
