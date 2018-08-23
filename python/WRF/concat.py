from datetime import timedelta
from . import *

class CCBase(WRFiles):

    outfile = Unicode('out.nc').tag(config=True)
    """Base name of the output netCDF file (**no extension**, numberings are added in case of restarts, in the form '_part#' where # is the number). Defaults to the :attr:`outfile` trait which can be set by command line ('-o' flag) and config."""

    interpolator = Unicode('scipy').tag(config=True)
    """Which interpolator (if any) to use: ``scipy`` - use :class:`~data.interpolate.GridInterpolator`; ``bilinear`` - use :class:`~data.interpolate.BilinearInterpolator`."""

    variables = Unicode().tag(config=True)
    """Name of variable(s) to extract."""

    interpolate = Bool(False).tag(config=True)
    """Whether or not to interpolate to station locations (see :class:`.Concatenator`)."""

    utc_delta = Instance(timedelta, kw={'hours': -4})
    """The offset from UTC to be applied to the concatenated data (assuming the original files are in UTC)."""

    lead_day = Integer(-1).tag(config=True)
    """Lead day of the forecast for which to search, if only one particular lead day is desired. (``-1`` denotes no particular lead day.)"""

    function = Unicode().tag(config=True)
    """Callable to be applied to the data before concatenation (after interpolation), in dotted from ('<module>.<function>')."""

    aliases = {'d': 'Concatenator.domain',
               'o': 'Concatenator.outfile',
               'v': 'Concatenator.variables'}

    flags = {'i': ({'Concatenator': {'interpolate': True}}, "interpolate to station locations")}

    @property
    def var_list(self):
        return [self.variables]

    def func(self, value):
        try:
            return self._func(value)
        except AttributeError:
            if self.function == '':
                self._func = lambda x: x
            else:
                mod, f = os.path.splitext(self.function)
                self._func = getattr(import_module(mod), f)
            return self._func(value)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None
