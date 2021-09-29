import copy
import os
import sys

from astropy.coordinates import SkyCoord
from astropy.table import Table, setdiff
from astropy import units as u

from astroquery.vizier import Vizier

module_path = os.path.abspath(os.path.join('..'))
orig_sys_path = copy.deepcopy(sys.path)
if module_path not in sys.path:
    sys.path.append(module_path)
import swasputils
sys.path = orig_sys_path


CATALOGUES = {
    'J/BaltA/9/646/catalog': {
        'keys': ('GCVS', 'RAdeg', 'DEJ2000', 'Per'),
        'coord_cols': ('RAdeg', 'DEJ2000'),
        'coord_units': (u.deg, u.deg),
        'mag_col': 'magMax',
    },
    'J/ApJS/247/44/table2': {
        'keys': ('_2MASS', 'FileName'),
        'coord_cols': ('RAJ2000', 'DEJ2000'),
        'coord_units': (u.hour, u.deg),
        'mag_col': 'Rmag',
    },
}

Vizier.ROW_LIMIT = -1

class Catalogue(object):
    @classmethod
    def all(cls):
        for catalogue, d in CATALOGUES.items():
            yield Catalogue(name=catalogue, **d)
    
    def __init__(self, name, keys, coord_cols, coord_units, mag_col):
        self.name = name
        self.keys = keys
        self.coord_cols = coord_cols
        self.coord_units = coord_units
        self.mag_col = mag_col
    
        self._full_table = None
        self._matched_table = None
        self._unmatched_table = None
        
        self._matched_coords = None
        self._unmatched_coords = None
        
    def __str__(self):
        return self.name
        
    @property
    def full_table(self):
        if not self._full_table:
            self._full_table =  Vizier.get_catalogs(self.name)[0]
        return self._full_table
    
    @property
    def full_coords(self):
        return self.coords(self.full_table)
    
    @property
    def full_mag(self):
        return self.mag(self.full_table)
    
    @property
    def matched_table(self):
        if not self._matched_table:
            catalogue_escaped = self.name.replace('/', '_')
            self._matched_table = Table.read(os.path.join(swasputils.CACHE_LOCATION, f'catalogue_match_{catalogue_escaped}.ecsv'))
        return self._matched_table
    
    @property
    def matched_coords(self):
        if not self._matched_coords:
            self._matched_coords = self.coords(self.matched_table)
        return self._matched_coords
    
    @property
    def matched_mag(self):
        return self.mag(self.matched_table)
    
    @property
    def unmatched_table(self):
        if not self._unmatched_table:
            self._unmatched_table = setdiff(self.full_table, self.matched_table, keys=self.keys)
        return self._unmatched_table
    
    @property
    def unmatched_coords(self):
        if not self._unmatched_coords:
            self._unmatched_coords = self.coords(self.unmatched_table)
        return self._unmatched_coords
    
    @property
    def unmatched_mag(self):
        return self.mag(self.unmatched_table)
    
    def coords(self, table):
        return SkyCoord(table[self.coord_cols[0]], table[self.coord_cols[1]], unit=self.coord_units)
    
    def coords_rad(self, coords=None):
        if not coords:
            coords = self.full_coords
        return (coords.ra.wrap_at(180 * u.deg).radian, coords.dec.radian)
    
    def mag(self, table):
        return table[self.mag_col]