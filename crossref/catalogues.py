import copy
import os
import sys

from astropy.coordinates import SkyCoord
from astropy.table import Table, setdiff, hstack, vstack
from astropy import units as u

from astroquery.vizier import Vizier

from IPython.display import clear_output

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
        
    @classmethod
    def coords_rad(self, coords):
        return (coords.ra.wrap_at(180 * u.deg).radian, coords.dec.radian)
        
    @classmethod
    def superwasp_sources(self):
        return Table.read(os.path.join(swasputils.CACHE_LOCATION, 'source_coords.fits'))
    
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
    
    def __repr__(self):
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
    def matching_table(self):
        """
        This generates a table suitable for cross-matching against other catalogues.
        The table contains all of this catalogue's key columns (prefixed with the catalogue name)
        plus the standard _RAJ2000 and _DEJ2000 columns for querying Vizier.
        """
        left_table = self.full_table[self.keys]
        for key in self.keys:
            left_table.rename_column(key, f'{self.name}:{key}')
        right_table = self.coords_table(self.full_coords)
        return hstack([left_table, right_table])
    
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
    
    def coords_table(self, coords):
        return Table(
            data=[
                coords.ra,
                coords.dec,
            ],
            names=[
                '_RAJ2000',
                '_DEJ2000',
            ],
            dtype=[
                'float64',
                'float64',
            ],
            units={
                '_RAJ2000': u.deg,
                '_DEJ2000': u.deg,
            }
        )
    
    def cross_match(self, other_catalogue=None, source_table=None):
        """
        Queries Vizier to cross match this catalogue with the other catalogue.
        """
        SPLIT_SIZE = 1000
        if not source_table:
            if not other_catalogue:
                raise RuntimeError('Need at least one of other_catalogue or source_table to cross match')
            source_table = other_catalogue.matching_table
        total_iterations = int(len(source_table) / SPLIT_SIZE) + 1
        
        results = []
        for i in range(total_iterations):
            clear_output(wait=True)
            print(f'Matching {self} against {other_catalogue}: {i + 1}/{total_iterations}')
            sources = source_table[i * SPLIT_SIZE : (i+1) * SPLIT_SIZE]
            if len(sources) > 0:
                try:
                    results.append(Vizier.query_region(sources, radius=2*u.arcmin, catalog=self.name)[0])
                except IndexError:
                    continue
                for key in source_table.columns.keys():
                    if key in ('_RAJ2000', '_DEJ2000'):
                        continue
                    results[-1].add_column(
                        [ sources[key][i - 1] for i in results[-1]['_q'] ],
                        name=key,
                    )
        if results:
            results = vstack(results)
            del results['_q']
        return results
    
    def mag(self, table):
        return table[self.mag_col]