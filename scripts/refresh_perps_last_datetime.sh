#!/bin/sh
psql -U quantfiction -d crypto_data -c 'REFRESH MATERIALIZED VIEW perps_last_datetime'