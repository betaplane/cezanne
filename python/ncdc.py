#!/usr/bin/env python
import requests, os, json, re
from bs4 import BeautifulSoup
import pandas as pd

base_url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/'


def parse_stations(record):
    rec = record.copy()
    full_id = rec.pop('id')
    num_id = re.search('(\d+)', full_id.split(':')[1]).group(1)
    cols, data = zip(*rec.items())
    row = pd.Series(data, index=cols, name=int(num_id)).to_frame().T
    row['id'] = full_id
    return row

def get(endpoint, parser=None, **params):
    with requests.Session() as s:
        s.headers.update({'token': 'OpqrLypwpgTRWdUZmjVkZKIFNsfovPSx'})
        s.params.update({'limit': 50})

        params.update({'offset': 1})
        tot = 1
        results = []
        while params['offset'] <= tot:
            r = s.get(os.path.join(base_url, endpoint), params=params)
            if not r.ok:
                print(r.text)
                break
            try:
                j = json.loads(r.text)
                tot = j['metadata']['resultset']['count']
                params.update({'offset': params['offset'] + 50})
                results.extend(j['results'])
            except:
                break
    if parser is None:
        return results
    else:
        return pd.concat([parser(r) for r in results], 0)


r = get('stations', extent='-90,-180,-50,180')

import BASdata as BAS
sfc = BAS.stations(BAS.sfc)
sfc['type'] = 'sfc'
# stations 89662, 89665 turn up in both lists
aws = BAS.stations(BAS.aws).drop([89662, 89665])
aws['type'] = 'aws'
s = pd.concat([sfc, aws])
s.loc[[89662, 89665], 'type'] = 'sfc+aws'

l = r.join(s, how='outer')

