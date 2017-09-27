#!/usr/bin/env python
import requests, os, json, re
from bs4 import BeautifulSoup
import pandas as pd

base_url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/'


def get(endpoint, **params):
    with requests.Session() as s:
        s.headers.update({'token': 'OpqrLypwpgTRWdUZmjVkZKIFNsfovPSx'})
        s.params.update({'limit': 50})

        params.update({'offset': 1})
        tot = 1
        results = []
        while params['offset'] <= tot:
            r = s.get(os.path.join(base_url, 'stations'), params=params)
            if not r.ok:
                break
            j = json.loads(r.text)
            tot = j['metadata']['resultset']['count']
            params.update({'offset': params['offset'] + 50})
            results.extend(j['results'])
    return results


r = get('stations', extent='-90,-180,-50,180')
ids = [int(re.search('(\d+)', x['id']).group(1)) for x in r]
