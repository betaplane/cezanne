#!/home/arno/Documents/code/conda/envs/misc/bin/python
"""
:mod:`traitlets.config`-based command-line app to search Zotero library, using `Pyzotero <https://pyzotero.readthedocs.io/en/latest/>`_. So far implemented: ``find`` subcommand which lists ``creatorSummary`` and collections for each item found.

Example use::

    ./zotero.py find scargle

As usual, to get help::

    ./zotero.py -h
    # or
    ./zotero.py --help-all

"""

import os
from traitlets.config import Application, Config
from traitlets.config.loader import PyFileConfigLoader, ConfigFileNotFound
from traitlets import Unicode, Integer
from pyzotero import zotero as zot

class base(Application):
    config_file = Unicode('~/Dropbox/work/config.py').tag(config=True)
    library_id = Integer().tag(config=True)
    key = Unicode().tag(config=True)

    def __init__(self, *args, config={}, **kwargs):
        try:
            cfg = PyFileConfigLoader(os.path.expanduser(self.config_file)).load_config()
            cfg.merge(config)
        except ConfigFileNotFound:
            cfg = Config(config)
        super().__init__(config=cfg, **kwargs)


class Find(base):

    info = Unicode('collection', 'What information to print for each found item.').tag(config=True)
    qmod = Unicode('titleCreatorYear',
                   "Query type: 'titleCreatorYear' (default) or 'everything' (option -e)").tag(config=True)
    flags = {'c': ({'Find': {'info': 'collection'}}, "get collection of items"),
             'e': ({'Find': {'qmod': 'everything'}}, "query mode 'everything'")}

    def start(self):
        # kwarg qmod={titleCreatorYear, everything}
        # https://www.zotero.org/support/dev/web_api/v3/basics
        q = self.parent.zot.items(q=self.extra_args[0])
        for i in q:
            try:
                getattr(self, self.info)(i)
            except Exception as e:
                print('{} -- ERROR: {}'.format(i['data']['title'], e))

    def collection(self, item):
        def r(key, s=[]):
            c = [i for i in self.parent.collections if i['key']==key][0]
            s.insert(0, c['data']['name'])
            p = c['data']['parentCollection']
            return s if p is False else r(p, s)
        c = ['/'.join(r(k)) for k in item['data']['collections']]
        print('{} is in {}'.format(item['meta']['creatorSummary'], '; '.join(c)))

class Zotero(base):

    subcommands = {'find': (Find, 'find a library item')}

    def initialize(self):
        self.parse_command_line()
        self.zot = zot.Zotero(self.library_id, 'user', self.key)

    @property
    def collections(self):
        if not hasattr(self, '_collections'):
            self._collections = self.zot.everything(self.zot.collections())
        return self._collections


if __name__ == '__main__':
    app = Zotero()
    app.initialize()
    app.subapp.start()
