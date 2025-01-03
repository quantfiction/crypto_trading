import pandas as pd, numpy as np
import requests
from urllib.parse import urljoin

from dotenv import dotenv_values, find_dotenv

# Credentials
config = dotenv_values(find_dotenv())
ad_api_key = config.get('AMBERDATA_API_KEY')

# Params
headers = {
    'accept':'application/json',
    'x-api-key':ad_api_key,
}
ad_base_url = 'https://api.amberdata.com'

def get_exchange_reference_futures(exchange, include_inactive=True):
    endpoint_exchange_reference = 'market/futures/exchanges/reference'
    url_exchange_reference = urljoin(ad_base_url, endpoint_exchange_reference)
    params = {
        'exchange': exchange,
        'includeInactive': 'true' if include_inactive else 'false',
    }

    with requests.Session() as session:
        session.headers.update(headers)

        list_instruments = []
        while url_exchange_reference is not None:
            response = session.get(url_exchange_reference, params=params)
            payload = response.json().get('payload', {})
            metadata = payload.get('metadata', {})
            data = payload.get('data', {})

            list_instruments.extend(
                pd.DataFrame(
                [{'instrument': instrument, **data_exchange[instrument]} for instrument in data_exchange.keys()]
            ).assign(exchange=exchange)
            for exchange, data_exchange in data.items()
        )

            url_exchange_reference = metadata.get('next', None)
    
    return pd.concat(list_instruments, axis=0)

def get_ohlcv_info_futures(exchange, include_inactive=True):
    endpoint_ohlcv_info = 'market/futures/ohlcv/information'
    url_ohlcv_info = urljoin(ad_base_url, endpoint_ohlcv_info)

    params = {
        'exchange':exchange,
        'includeInactive':'true' if include_inactive else 'false',
    }

    with requests.Session() as session:
        session.headers.update(headers)

        list_instruments = []
        while url_ohlcv_info is not None:
            response = session.get(url_ohlcv_info, params=params)
            payload = response.json().get('payload', {})
            metadata = payload.get('metadata', {})
            data = payload.get('data', {})

            list_instruments.append(pd.DataFrame(data))

            url_ohlcv_info = metadata.get('next', None)

    return pd.concat(list_instruments, axis=0)
            


