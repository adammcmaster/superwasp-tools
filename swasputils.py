import functools
import multiprocessing
import os
import pathlib
import shelve

import numpy
import pandas
import seaborn
import ujson
import urllib

from collections import defaultdict 

from IPython.display import Image, display

from astropy.coordinates import SkyCoord
from astropy import units as u
import astropy.io.fits as fits
import astropy.utils.data
from astropy.table import vstack
from astropy.timeseries import TimeSeries
from astropy.stats import sigma_clip

from astroquery.vizier import Vizier

from matplotlib import pyplot

astropy.utils.data.Conf.remote_timeout.set(60)


DATA_LOCATION = os.path.join('..', '..', 'superwasp-data')
CACHE_LOCATION = os.path.join(DATA_LOCATION, 'cache')

SECONDS_PER_DAY = 60 * 60 * 24

MAIN_WORKFLOW = 7534
JUNK_WORKFLOW = 17313

if not os.path.exists(DATA_LOCATION):
    os.mkdir(DATA_LOCATION)
if not os.path.exists(CACHE_LOCATION):
    os.mkdir(CACHE_LOCATION)


def batches(i, batch_size=100):
    for x in range(int(len(i)/batch_size)+1):
        subset = i[x*batch_size:(x+1)*batch_size]
        if len(subset) == 0:
            return
        yield subset


class PandasDFWrapper(object):
    def limit(self, limit):
        return self.__class__(df=self.df[:limit])
    
    def cached_pandas_load(self, filename):
        cache_file_path = pathlib.Path(os.path.join(CACHE_LOCATION, '{}.pickle'.format(filename)))
        orig_file_path = pathlib.Path(os.path.join(DATA_LOCATION, filename))
        if cache_file_path.exists() and (
            not orig_file_path.exists() or
            cache_file_path.stat().st_mtime > orig_file_path.stat().st_mtime
        ):
            return (pandas.read_pickle(cache_file_path), cache_file_path)
        return (None, cache_file_path)

    def _mpapply(self, func, df=None, axis='rows'):
        """
        Splits the DataFrame and applies the function across a pool of worker processes.

        On experiment this can actually add a lot of overhead to execution time.
        But it might be worth it on really long running operations.
        """
        if df is None:
            df = self.df

        split_df = numpy.array_split(df, multiprocessing.cpu_count())

        results = []
        with multiprocessing.Pool() as pool:
            for part in split_df:
                results.append(pool.apply_async(part.apply, args=(func, axis)))
            return pandas.concat([r.get() for r in results])


class CoordinatesMixin(object):
    VSX_MAG_AMPLITUDE_FLAG = '('
    VSX_PERIOD_THRESHOLD = 0.1
    VSX_SEARCH_RADIUS = 2 * u.arcsec
    VSX_MAGNITUDE_LIMIT = 15
    
    @property
    def coords(self):
        return SkyCoord(self.df['SWASP ID'].replace(r'^1SWASP', '', regex=True).values, unit=(u.hour, u.deg))
    
    @property
    def fits_urls(self):
        return self.df['SWASP ID'].apply(lambda s: 'http://wasp.warwick.ac.uk/lcextract?{}'.format(
            urllib.parse.urlencode(
                {'objid': s.replace('1SWASP', '1SWASP ')},
                quote_via=urllib.parse.quote,
            )
        ))
    
    @property
    def fits(self):
        for swasp_id, url in zip(self.df['SWASP ID'], self.fits_urls):
            yield fits.open(url)
            
    @property
    def timeseries(self):
        for fits_file in self.fits:
            hjd_col = fits.Column(name='HJD', format='D', array=fits_file[1].data['TMID']/86400 + 2453005.5)
            lc_data = fits.BinTableHDU.from_columns(fits_file[1].data.columns + fits.ColDefs([hjd_col]))
            yield TimeSeries.read(lc_data, time_column='HJD', time_format='jd')
            
    @property
    def timeseries_folded(self):
        for period, timeseries in zip(self.df['Period'], self.timeseries):
            yield timeseries.fold(
                period=period * u.second,
                normalize_phase=False,
            )
    
    def add_coords(self):
        if 'Coords' not in self.df:
            coords = self.coords
            self.df['_RAJ2000'] = coords.ra
            self.df['_DEJ2000'] = coords.dec
            
    def add_fits_urls(self):
        if 'FITS URL' not in self.df:
            self.df['FITS URL'] = self.fits_urls
    
    def _extend_epochs(self, ts, epochs=1):
        epoch_length = ts['time'].max() - ts['time'].min()
        ts_out = [ts]
        for i in range(epochs):
            ts_new = ts.copy()
            ts_new['time'] = ts_new['time'] + epoch_length * (i + 1)
            ts_out.append(ts_new)
        return vstack(ts_out)

    def plot(self, folded=False, clip=False, sigma=4, hue=None):
        if folded:
            self.add_classification_labels()
            if 'Period' not in self.df:
                self.df = self.df.merge(FoldedLightcurves().df, how='left')
            ts_iter = self.timeseries_folded
        else:
            plotted_ids = set()
            ts_iter = self.timeseries

        for (subject_id, row), ts in zip(self.df.iterrows(), ts_iter):
            if folded:
                ts = self._extend_epochs(ts)
            else:
                if row['SWASP ID'] in plotted_ids:
                    continue
                plotted_ids.add(row['SWASP ID'])

            if clip:
                ts_flux = sigma_clip(ts['TAMFLUX2'], sigma=sigma)
            else:
                ts_flux = ts['TAMFLUX2']
            
            ts_data = {
                'time': ts.time.jd,
                'flux': ts_flux,
                'camera': ts['CAMERA_ID'],
            }
            pyplot.figure()
            plot = seaborn.scatterplot(
                data=ts_data,
                x='time',
                y='flux',
                hue=hue,
                alpha=0.5,
                s=1,
                palette='Set2',
            )
            if folded:
                plot.set_title('{} Period {}s ({})'.format(
                    row['SWASP ID'],
                    row['Period'],
                    row['Classification Label'],
                ))
            else:
                plot.set_title(row['SWASP ID'])
        
    def _query_vsx_for_coord(self, coord, cache):
        coord_str = coord.to_string()
        if coord_str not in cache:
            cache[coord_str] = Vizier.query_region(
                coord,
                radius=self.VSX_SEARCH_RADIUS, 
                catalog='B/vsx/vsx',
            )

        return cache[coord_str]
    
    def _coords_for_row(self, row, cache):
        if row['SWASP ID'] not in cache:
            cache[row['SWASP ID']] = SkyCoord(
                row['SWASP ID'].replace('1SWASP', ''),
                unit=(u.hour, u.deg)
            )
        
        return cache[row['SWASP ID']]
 
    def add_vsx_types(self):
        if self.df.index.name:
            orig_index_name = self.df.index.name
            self.df.reset_index(inplace=True)
        else:
            orig_index_name = None

        vsx_types, vsx_types_cache_file = self.cached_pandas_load('vsx_types')
        if vsx_types is None:
            vsx_results_dict = defaultdict(list)
            batch_size = 100
            result_map = {
                'VSX Period': 'Period',
                'VSX Type': 'Type',
                'VSX Name': 'Name',
                'VSX Mag Max': 'max',
                'VSX Mag Min': 'min',
                'VSX Mag Format': 'f_min',
            }

            with shelve.open(os.path.join(CACHE_LOCATION, 'vsx_cache')) as vsx_cache:
                with shelve.open(os.path.join(CACHE_LOCATION, 'coord_cache')) as coord_cache:
                    for i, (_, row) in enumerate(self.df.iterrows(), start=1):
                        if i % 100 == 0:
                            print('Processing row: {}'.format(i), end='\r')
                        vsx_query = self._query_vsx_for_coord(
                            self._coords_for_row(row, coord_cache),
                            vsx_cache
                        )
                        if vsx_query is None:
                            continue

                        period_min = (row['Period'] / SECONDS_PER_DAY) * (1 - self.VSX_PERIOD_THRESHOLD)
                        period_max = (row['Period'] / SECONDS_PER_DAY) * (1 + self.VSX_PERIOD_THRESHOLD)

                        for vsx_table in vsx_query:
                            for vsx_row in vsx_table:
                                if vsx_row['Period'] < period_min:
                                    continue
                                if vsx_row['Period'] > period_max:
                                    continue
                                
                                # When max is actually a mean and min is actually an amplitude
                                if vsx_row['f_min'] == self.VSX_MAG_AMPLITUDE_FLAG:
                                    if (vsx_row['max'] + vsx_row['min']) > self.VSX_MAGNITUDE_LIMIT:
                                        continue
                                else:
                                    if vsx_row['min'] > self.VSX_MAGNITUDE_LIMIT:
                                        continue

                                vsx_results_dict['subject_id'].append(row['subject_id'])
                                for result_key, vsx_key in result_map.items():
                                    vsx_results_dict[result_key].append(vsx_row[vsx_key])

            vsx_types = pandas.DataFrame(vsx_results_dict)
            vsx_types.to_pickle(vsx_types_cache_file)

        if len(vsx_types.index) > 0:
            vsx_types['VSX Period'] = vsx_types['VSX Period'] * SECONDS_PER_DAY
            self.df = self.df.merge(
                vsx_types,
                left_on='subject_id',
                right_on='subject_id',
                how='left',
            )

        if orig_index_name:
            self.df.set_index('subject_id', inplace=True)

        
class ZooLookupMixin(object):
    @property
    def zoo_lookup(self):
        zoo_lookup, cache_file = self.cached_pandas_load('lookup.dat')
        if zoo_lookup is not None:
            return zoo_lookup
        
        zoo_lookup = pandas.read_csv(
            os.path.join(DATA_LOCATION, 'lookup.dat'),
            delim_whitespace=True,
            header=None,
        )
        zoo_lookup.columns = [
            'subject_id',
            'SWASP ID',
            'Period',
            'Period Number',
        ]
        # Period in this file is rounded differently to the others
        # So drop it here so it doesn't stop us from merging later
        zoo_lookup.drop('Period', 'columns', inplace=True)
        zoo_lookup.to_pickle(cache_file)
        return zoo_lookup
    
    def merge_zoo_lookup(self):
        if self.df.index.name:
            orig_index_name = self.df.index.name
            self.df.reset_index(inplace=True)
        else:
            orig_index_name = None
                
        self.df = self.df.merge(
            self.zoo_lookup,
            how='left',
        )
        if orig_index_name:
            self.df.set_index(orig_index_name, inplace=True)


class ZooniverseSubjects(PandasDFWrapper, ZooLookupMixin):
    def __init__(self, df=None):
        if df is not None:
            self.df = df
            return

        self.df, self.cache_file = self.cached_pandas_load('superwasp-variable-stars-subjects.csv')
        if self.df is not None:
            return
        
        self.df = pandas.read_csv(
            os.path.join(DATA_LOCATION, 'superwasp-variable-stars-subjects.csv'),
            index_col='subject_id',
        )
        self.df.to_pickle(self.cache_file)
    
    @property
    def subject_sets(self):
        return { set_id: self.get_subject_set(set_id) for set_id in set(self.df['subject_set_id']) }
    
    @property
    def workflows(self):
        return { 
            workflow_id: self.get_workflow(workflow_id) 
            for workflow_id in set(self.df[self.df['workflow_id'].notna()]['workflow_id'])
        }

    @property
    def retired(self):
        return self.__class__(df=self.df[self.df['retired_at'].notna()])

    @property
    def active(self):
        return self.__class__(df=self.df[self.df['retired_at'].isna()])
    
    @property
    def distinct(self):
        new_df = self.df.reset_index('subject_id')
        new_df.drop_duplicates('subject_id', inplace=True)
        new_df.set_index('subject_id', inplace=True)
        return self.__class__(df=new_df)
    
    def get_subject_set(self, set_id):
        return self.__class__(df=self.df[self.df['subject_set_id'] == set_id])
    
    def get_workflow(self, workflow_id):
        return self.__class__(df=self.df[self.df['workflow_id'] == workflow_id])
    
    def decode_locations(self, index=0, target='lightcurve'):
        self.df = self.df.copy()
        self.df[target] = self.df['locations'].apply(
            lambda s: ujson.loads(s)[str(index)]
        )
    
    def display_lightcurves(self, col='lightcurve', start=0, end=None):
        if col not in self.df:
            self.decode_locations(target=col)
            
        self.df[col][start:end].apply(
            lambda s: display(Image(url=s, width=500, height=500))
        )


class ZooniverseClassifications(PandasDFWrapper):
    ANNOTATION_PREFIX = 'annotation_'
    
    def __init__(self, df=None, drop_duplicates=False, duplicate_columns=('subject_ids', 'user_id')):
        if df is not None:
            self.df = df
            return

        try:
            self.df, self.cache_file = self.cached_pandas_load('superwasp-variable-stars-classifications.csv')
            if self.df is not None:
                return

            self.df = pandas.read_csv(
                os.path.join(DATA_LOCATION, 'superwasp-variable-stars-classifications.csv'),
                index_col='classification_id',
            )
            self.df.to_pickle(self.cache_file)
        finally:
            if drop_duplicates:
                self.df.drop_duplicates(duplicate_columns, inplace=True)
    
    @property
    def workflows(self):
        return { 
            workflow_id: self.get_workflow(workflow_id) 
            for workflow_id in set(self.df[self.df['workflow_id'].notna()]['workflow_id'])
        }
    
    @property
    def annotations(self):
        self.decode_annotations()
        return self.df[['subject_ids', 'user_id'] + self.annotation_keys]
    
    @property
    def annotation_keys(self):
        self.decode_annotations()
        return [col for col in self.df.keys() if col.startswith(self.ANNOTATION_PREFIX)]
    
    def get_workflow(self, workflow_id):
        return ZooniverseClassifications(df=self.df[self.df['workflow_id'] == workflow_id])
    
    def get_subjects(self, subject_ids):
        return ZooniverseClassifications(df=self.df[self.df['subject_ids'].isin(subject_ids)])
    
    def get_users(self, user_names):
        return ZooniverseClassifications(df=self.df[self.df['user_name'].isin(user_names)])

    def decode_annotations(self):
        if not 'annotations' in self.df.keys():
            return
        self.df = self.df.copy()
        
        for classification_id, annotations in self.df['annotations'].items():
            for annotation in ujson.loads(annotations):
                annotation_col = self.ANNOTATION_PREFIX + annotation['task']
                if annotation_col not in self.df:
                    self.df[annotation_col] = pandas.Series([], dtype=str)
                self.df.at[classification_id, annotation_col] = annotation['value']
        self.df.drop('annotations', 'columns', inplace=True)
    
    def count_annotations(self, col=None, drop_duplicates=True):
        self.decode_annotations()
        if not col:
            col = self.annotation_keys[0]

        df = self.annotations.reset_index()
        if drop_duplicates:
            df.drop_duplicates(['user_id', 'subject_ids'], inplace=True)

        return pandas.pivot_table(
            df, 
            index='subject_ids', 
            values='classification_id', 
            columns=col,
            aggfunc=lambda x: len(x.unique()),
            fill_value=0,
        )


class FoldedLightcurves(PandasDFWrapper, CoordinatesMixin, ZooLookupMixin):
    def __init__(self, min_period=0, df=None):
        self.min_period = min_period
        
        if df is not None:
            self.df = df
            return
        
        self.df, self.cache_file = self.cached_pandas_load('results_total.dat')
        if self.df is not None:
            return
        
        self.df = pandas.read_csv(
            os.path.join(DATA_LOCATION, 'results_total.dat'),
            delim_whitespace=True,
            header=None,
        )
        self.df.columns = [
            'Camera Number',
            'SWASP',
            'ID',
            'Period Number',
            'Period',
            'Sigma',
            'Chi Squared',
            'Period Flag'
        ]
        self.df = self.df[(self.df['Period Flag'] == 0) & (self.df['Period'] >= min_period)]
        self.df['SWASP ID'] = self.df['SWASP'] + self.df['ID']
        self.df.drop(['Period Flag', 'Camera Number', 'SWASP', 'ID'], 'columns', inplace=True)
        self.df.to_pickle(self.cache_file)

    def get_siblings(self, swasp_id):
        return self.__class__(df=self.df[self.df['SWASP ID'] == swasp_id], min_period=self.min_period)


class AggregatedClassifications(PandasDFWrapper, CoordinatesMixin):
    PULSATOR = 1
    EA_EB = 2
    EW = 3
    ROTATOR = 4
    UNKNOWN = 5
    JUNK = 6
    CLASSIFICATION_LABELS = {
        PULSATOR: 'Pulsator',
        EA_EB: 'EA/EB',
        EW: 'EW',
        ROTATOR: 'Rotator',
        UNKNOWN: 'Unknown',
        JUNK: 'Junk',
    }

    def __init__(self, df=None):
        if df is not None:
            self.df = df
            return
        
        self.df, self.cache_file = self.cached_pandas_load('class_top.csv')
        if self.df is not None:
            return

        self.df = pandas.read_csv(
            os.path.join(DATA_LOCATION, 'class_top.csv'),
            delim_whitespace=True,
            header=None,
        )
        self.df.columns = [
            'subject_id',
            'SWASP ID',
            'Period Number',
            'Period',
            'Classification',
            'Period Uncertainty',
            'Classification Count',
        ]
        # Period in this file is rounded differently to the others
        # So drop it here so it doesn't stop us from merging later
        self.df.drop('Period', 'columns', inplace=True)
        self.df.set_index('subject_id', inplace=True)
        self.df.to_pickle(self.cache_file)

    def add_classification_labels(self):
        self.df = self.df.copy()
        self.df['Classification Label'] = self.get_classification_labels(self.df['Classification'])

    def get_classification_labels(self, series):
        return series.apply(lambda c: self.CLASSIFICATION_LABELS.get(c, None))

    def get_class(self, classification):
        return self.__class__(df=self.df[self.df['Classification'] == classification])

    def remove_class(self, classification):
        return self.__class__(df=self.df[self.df['Classification'] != classification])

    @property
    def pulsators(self):
        return self.get_class(self.PULSATOR)

    @property
    def eaebs(self):
        return self.get_class(self.EA_EB)

    @property
    def ews(self):
        return self.get_class(self.EW)

    @property
    def rotators(self):
        return self.get_class(self.ROTATOR)

    @property
    def unknowns(self):
        return self.get_class(self.UNKNOWN)

    @property
    def junk(self):
        return self.get_class(self.JUNK)

    @property
    def real(self):
        return self.remove_class(self.JUNK)


class UnifiedSubjects(ZooniverseSubjects, FoldedLightcurves, AggregatedClassifications):
    def __init__(
        self, 
        zooniverse_subjects=None, 
        folded_lightcurves=None,
        aggregated_classifications=None,
        df=None, 
        min_period=0
    ):
        self.min_period = min_period
        
        if df is not None:
            self.df = df
            return
        
        if not zooniverse_subjects:
            zooniverse_subjects = ZooniverseSubjects()
        if not folded_lightcurves:
            folded_lightcurves = FoldedLightcurves(min_period=min_period)
        if not aggregated_classifications:
            aggregated_classifications = AggregatedClassifications()
        
        self.df = zooniverse_subjects.df
        self.merge_zoo_lookup()
        self.df = self.df.reset_index().merge(
            aggregated_classifications.df.reset_index(),
            how='left',
        )
        self.df = self.df.merge(
            folded_lightcurves.df,
            how='left',
        )
        self.df.set_index('subject_id', inplace=True)
    
    def get_siblings(self, obj_id):
        if type(obj_id) == int:
            swasp_id = self.df[self.df.index == obj_id].iloc[0]['SWASP ID']
        else:
            swasp_id = obj_id
        
        return super().get_siblings(swasp_id)
    

