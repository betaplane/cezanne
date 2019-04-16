#!/usr/bin/env python
"""
Base classes
============

This module provides :mod:`traitlets` based configurable classes which are intended to be used by all my other modules / packages. The rationale is that if I discover an error in my logic, I only need to change it here.

It also means that whenever I factor out some package from the rest of my code, I might need to include this module in the package.

Notes
-----
* :mod:`traitlets` initializes parent or child classes sometimes with the ``config`` keyword, which takes a :class:`traitlets.config.Config` instance.
* It doesn't seem feasible to maintain a reasonable order of precedence as to the sources from which traits are populated **and** make ``config_file`` a trait itself which can possibly be set from the command line.
* I assume the following order of precedence:
    * init arguments supersede config files
    * command-line arguments supersede init arguments

"""

from traitlets.config import Application, Configurable, Config
from traitlets.config.loader import PyFileConfigLoader, ConfigFileNotFound
from traitlets import Unicode

class Conf(Configurable):
    config_file = '~/.cezanne_config.py'
    def __init__(self, config={}, **kwargs):
        try:
            cfg = PyFileConfigLoader(self.config_file).load_config()
            cfg.merge(config)
            super().__init__(config=cfg, **kwargs)
        except ConfigFileNotFound:
            print('not found')
            super().__init__(config=Config(config), **kwargs)

class App(Conf, Application):
    def __init__(self, **kwargs):
        super(App, self).__init__(**kwargs)
        self.parse_command_line()
