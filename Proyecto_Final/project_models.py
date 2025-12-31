
import numpy as np
import pandas as pd
from prophet import Prophet

def preparar_diario(df, canal, col_fecha='Fecha_Documento_Factura', col_y='Cantidad'):
    s = (df.loc[df['Canal_Venta'] == canal, [col_fecha, col_y]]
            .groupby(col_fecha, as_index=False).sum()
            .rename(columns={col_fecha:'ds', col_y:'y'}))
    s['ds'] = pd.to_datetime(s['ds'])
    s = s.sort_values('ds').reset_index(drop=True)
    return s

def split_weekly(dfw, test_weeks=8):
    return dfw.iloc[:-test_weeks].copy(), dfw.iloc[-test_weeks:].copy()

def weekly_with_gap_regressor(df_daily, doy_event_set, week_rule='W-SUN'):
    d = df_daily[['ds','y']].copy()
    d['ds'] = pd.to_datetime(d['ds'])
    d['doy'] = d['ds'].dt.dayofyear
    d['is_gap_day'] = d['doy'].isin(doy_event_set).astype(int)

    # semana
    d = d.set_index('ds')
    w = d.resample(week_rule).agg(y=('y','sum'), gap_ratio=('is_gap_day','mean')).reset_index()
    w = w.rename(columns={'ds':'ds'})
    # suavizado
    w['gap_smooth'] = w['gap_ratio'].rolling(3, min_periods=1, center=True).mean()
    w['is_event_week'] = (w['gap_ratio'] > 0).astype(int)
    return w[['ds','y','gap_ratio','gap_smooth','is_event_week']]

def prophet_weekly_with_gap_and_events(train_w_ext, test_w_ext):
    mw = Prophet(
        growth='linear',
        weekly_seasonality=False,
        yearly_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=8
    )
    mw.add_seasonality(name='yearly_custom', period=365.25, fourier_order=10)

    mw.add_regressor('gap_smooth', mode='additive', prior_scale=10)
    mw.add_regressor('is_event_week', mode='additive', prior_scale=30)

    mw.add_country_holidays('AR')
    mw.fit(train_w_ext[['ds','y','gap_smooth','is_event_week']])

    fw = test_w_ext[['ds','gap_smooth','is_event_week']].copy()
    fc = mw.predict(fw)
    yhat = np.clip(fc['yhat'].values, 0, None)
    return mw, yhat
