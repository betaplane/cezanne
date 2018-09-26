import os, requests, re
from datetime import datetime
from traitlets.config import Application
from traitlets.config.loader import PyFileConfigLoader, ConfigFileNotFound
from traitlets import Unicode
from bs4 import BeautifulSoup


# https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+Python
class SessionWithHeaderRedirection(requests.Session):
    AUTH_HOST = 'urs.earthdata.nasa.gov'
    def __init__(self, username, password):
        super().__init__()
        self.auth = (username, password)

    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url

        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

            if (original_parsed.hostname != redirect_parsed.hostname) and \
                    redirect_parsed.hostname != self.AUTH_HOST and \
                    original_parsed.hostname != self.AUTH_HOST:
                del headers['Authorization']
        return

class EarthData(Application):
    config_file = Unicode('~/Dropbox/work/config.py').tag(config=True)
    def __init__(self):
        config = PyFileConfigLoader(os.path.expanduser(self.config_file)).load_config()
        super().__init__(config=config)

class SMAP(EarthData):
    def __init__(self):
        super().__init__()
        self.session = SessionWithHeaderRedirection(self.username, self.password)

    def listdir(self, url):
        r = self.session.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        links = [n.get('href') for n in soup.find_all('a')]

    def get(self, url = 'https://n5eil01u.ecs.nsidc.org/SMAP/SPL3SMP.005/',
            from_date = datetime(2015, 8, 15),
            to_date = datetime.now().date()):
        links = [l for l in self.listdir(url) if re.match('\d{4}\.\d\d\.\d\d', l)]
        for l in set(links):
            d = datetime.strptime(l[:10], '%Y.%m.%d')
            if d < from_date or d > to_date:
                continue
            h5 = [h for h in self.listdir(url + l) if h[-2:] == 'h5'][0]

    def write_file(self, filename, resp):
        with open(filename, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=1024**2):
                f.write(chunk)

if __name__ == '__main__':
    pass
