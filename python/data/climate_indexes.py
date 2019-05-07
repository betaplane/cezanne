"""
Data Webcrawler
===============


"""
#!/usr/bin/env python
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import re, requests, os
from bs4 import BeautifulSoup as BS
from urllib import parse as up
from ftplib import FTP
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor as Exec
from collections import namedtuple
import webbrowser


class Links:
    parser_args = {
        # initial args for bs4
        'kwargs': {'href': re.compile('^(?!#).*(?<!\.nc)(?<!\.css)$')}, # no starting '#', no ending in '.nc', '.css'
        # args to be applied against the 'links' dataframe
        'links': {
            'path': re.compile('^/?[^/]+'),  # not only a '/'
            'query': re.compile('^$')        # only empty query
        },
        # filter out stuff (e.g. binaries) before downloading it (called while following a link)
        'headers': {
            'Content-Type': re.compile('^(?!application)') # doesn't start with 'application' (e.g. netcdf files)
        }
    }

    urls = [
        'https://climatedataguide.ucar.edu/climate-data/overview-climate-indices',
        'https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni',
        'https://www.esrl.noaa.gov/psd/enso/',
        'https://climatedataguide.ucar.edu/climate-data/southern-oscillation-indices-signal-noise-and-tahitidarwin-slp-soi'
    ]

    response = namedtuple('response', ['text', 'ok', 'headers'])
    max_workers = 20

    def __init__(self, url=None, copy=None):
        if copy is not None:
            self.__dict__ = copy.__dict__
        else:
            if isinstance(url, int):
                url = self.urls[url]
                print('Fetching {}'.format(url))
            r = requests.get(url)
            r.raise_for_status()
            self._html = r.text
            self.url = url
            self._split_url = up.urlsplit(url)
            self._loaded_text = {}
            self.parse_page()

    def parse_page(self, *args, **kwargs):
        if kwargs == {}:
            kwargs = self.parser_args['kwargs']
        bs = BS(self._html, 'html5lib')
        links = pd.DataFrame([self._parse_tag(tag) for tag in bs.find_all(*args, **kwargs)])
        for k, v in self.parser_args['links'].items():
            links = links[links[k].str.contains(v)]
        _, i = np.unique(links['href'], return_index=True)
        self.links = links.iloc[i].reset_index(drop=True)

    def crawl_brute(self, headers_only=False):
        links = self.links['href'].dropna()
        prog = tqdm(total=len(links))
        def f(url):
            url = up.urljoin(self.url, url)
            ut = up.urlsplit(url)
            if ut.scheme=='ftp':
                r = self.get_ftp(ut)
            else:
                r = requests.head(url)
                if not headers_only:
                    if all([v.search(r.headers[k]) for k, v in self.parser_args['headers'].items()]):
                        r = requests.get(url)
                    prog.update(1)
            return r
        with Exec(max_workers=self.max_workers) as ex:
            reqs = list(ex.map(f, links))
            if not headers_only:
                self._loaded_text = {k: v.text for k, v in zip(links.index, reqs) if v.ok}
            self.headers = pd.DataFrame([dict(r.headers.items()) if r.ok else {} for r in reqs], index=links.index)

    def process(self):
        self._success, self._fail = {}, []
        for k, v in tqdm(self._loaded_text.items()):
            url = up.urljoin(self.url, self.links.loc[k]['href'])
            x = {}
            for c in self.__class__.__subclasses__():
                try:
                    x.update({c.__name__: c.data(StringIO(v) if isinstance(v, str) else v)})
                except: pass
                if hasattr(c, 'url_match') and re.search(c.url_match, url):
                    try:
                        x = {c.__name__: x[c.__name__]}
                        break
                    except:
                        print(c.__name__, k) # for debugging
                        raise
            if len(x) == 0:
                self._fail.append(k)
            else:
                self._success[k] = x
        print('')
        print('successes: {}'.format(len(self._success)))
        print('failures:  {}'.format(len(self._fail)))

    @property
    def successes(self):
        c = pd.Series({k: v.keys() for k, v in self._success.items()}, name='klass')
        df = self.links.loc[self._success.keys(), 'text']
        return pd.concat((df, c), 1)

    @property
    def failures(self):
        return self.links.loc[self._fail, 'text']

    @property
    def plaintext(self):
        content = self.headers['Content-Type'].dropna()
        idx = content.index[content.str.contains('text/plain')]
        return self.links.loc[idx]['text']

    @property
    def ftp(self):
        return self.links[self.links['scheme']=='ftp']['text']


    def get_buffer(self, n):
        t = self._loaded_text[n]
        return StringIO(t) if isinstance(t, str) else t

    @staticmethod
    def _parse_tag(tag):
        href = tag.attrs['href']
        d = dict(zip(['scheme', 'netloc', 'path', 'query', 'fragment'], up.urlsplit(href)))
        d.update(text=tag.text, href=href)
        return d

    def __getitem__(self, n):
        if hasattr(n, '__iter__'):
            return [up.urljoin(self.url, self.links.iloc[i]['href']) for i in n]
        return up.urljoin(self.url, self.links.iloc[n]['href'])

    def open(self, n):
        url = up.urljoin(self.url, self.links.iloc[n]['href'])
        webbrowser.open(url)

    def get_ftp(self, urltuple):
        url = urltuple.geturl()
        try:
            return self.response(self._loaded_text[url], True)
        except:
            try:
                ftp = FTP(urltuple.netloc)
                ftp.login()
                b = BytesIO()
                ftp.retrbinary('RETR {}'.format(urltuple.path), lambda c: b.write(c))
                ftp.close()
                b.seek(0)
                self._loaded_text[url] = b
                return self.response(b, True, {})
            except:
                return self.response('', False, {})

    @staticmethod
    def skipfooter(buf, pattern):
        try:
            while not re.match(pattern, next(buf)):
                continue
            skip = len(buf.readlines()) + 1
        except:
            skip = 0
        finally:
            buf.seek(0)
        return skip

    @staticmethod
    def skipheader(buf):
        posA = 0
        prev = next(buf)
        while True:
            posB = buf.seek(0, 1)
            n = next(buf)
            if len(n) == len(prev):
                break
            prev = n
            posA = posB
        buf.seek(posA)


class UCAR(Links):
    url_match = 'climatedataguide.ucar.edu/sites/default/files'

    @classmethod
    def data(cls, buf, skiprows=0, skipfooter=0):
        df = pd.read_csv(buf,
                         delim_whitespace=True,
                         skiprows=skiprows,
                         skipfooter=skipfooter,
                         index_col=0,
                         na_values=-99.99,
                         header=None).stack()
        df.index = pd.DatetimeIndex(['{}-{}'.format(*i) for i in df.index.tolist()])
        return df

class NCEP(Links):
    url_match = 'ftp.cpc.ncep.noaa.gov'

    @classmethod
    def data(cls, buf):
        # buf is BytesIO
        heads, data = [BytesIO()], [BytesIO()]
        pattern = re.compile(b'^\d{4}')
        buf.seek(0)
        try:
            while True:
                l = next(buf)
                if pattern.search(l):
                    data[-1].write(l)
                    if heads[-1].seek(0, 1) != 0:
                        heads.append(BytesIO())
                else:
                    heads[-1].write(l)
                    if data[-1].seek(0, 1) != 0:
                        data.append(BytesIO())
        except StopIteration:
            pass

        def f(*args):
            i, (h, d) = args
            d.seek(0)
            x = UCAR.data(d)
            return {'data': x, 'head': h.getvalue().decode()}

        return [f(*d) for d in enumerate(zip(heads[:len(data)], data))]

class ESRL(Links):
    url_match = 'www.esrl.noaa.gov/psd'

    @classmethod
    def data(cls, buf, skipheader=0):
        skipfooter = cls.skipfooter(buf, ' *-99.9')
        return UCAR.data(buf, skipheader, skipfooter)

