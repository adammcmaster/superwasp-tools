import os

import pandas
import ujson

from IPython.display import Image, display
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier


DATA_LOCATION = os.path.join('..', '..', 'superwasp-data')
SECONDS_PER_DAY = 60 * 60 * 24


class CoordinatesMixin(object):
    VSX_MAG_AMPLITUDE_FLAG = '('
    
    @property
    def coords(self):
        return SkyCoord(self.df['SWASP ID'].replace(r'^1SWASP', '', regex=True).values, unit=(u.hour, u.deg))
    
    def add_coords(self):
        self.df['Coords'] = self.coords
    
    def _get_vsx_types_for_row(self, row):
        PERIOD_THRESHOLD = 0.1
        SEARCH_RADIUS = 2 * u.arcsec
        MAGNITUDE_LIMIT = 15
        
        vsx_query = Vizier.query_region(
            row['Coords'],
            radius=SEARCH_RADIUS, 
            catalog='B/vsx/vsx',
        )
        
        period_min = row['Period'] / SECONDS_PER_DAY * (1 - PERIOD_THRESHOLD)
        period_max = row['Period'] / SECONDS_PER_DAY * (1 + PERIOD_THRESHOLD)
        
        result_map = {
            'VSX Period': 'Period',
            'VSX Type': 'Type',
            'VSX Name': 'Name',
            'VSX Mag Max': 'max',
            'VSX Mag Min': 'min',
            'VSX Mag Format': 'f_min',
        }
        
        results = {
            'subject_id': [],
        }
        results.update({k: [] for k in result_map})

        for vsx_table in vsx_query:
            vsx_df = vsx_table.to_pandas()
            
            matching_entries = vsx_df[
                (vsx_df['Period'] >= period_min) &
                (vsx_df['Period'] <= period_max) &
                (
                    ( # When max is actually a mean and min is actually an amplitude
                        (vsx_df['f_min'] == self.VSX_MAG_AMPLITUDE_FLAG) &
                        ((vsx_df['max'] + vsx_df['min']) <= MAGNITUDE_LIMIT)
                    ) |
                    ( # When max and min are actually what their names imply
                        (vsx_df['f_min'] != self.VSX_MAG_AMPLITUDE_FLAG) &
                        ((vsx_df['min']) <= MAGNITUDE_LIMIT)
                    )
                )
            ]
            
            for index, vsx_row in matching_entries.iterrows():
                results['subject_id'].append(row['subject_id'])
                for result_key, vsx_key in result_map.items():
                    results[result_key].append(vsx_row[vsx_key])
 
        return results

    def add_vsx_types(self):
        self.add_coords()
        if self.df.index.name:
            orig_index_name = self.df.index.name
            self.df.reset_index(inplace=True)
        else:
            orig_index_name = None
        vsx_results = self.df.apply(
            lambda r: self._get_vsx_types_for_row(r),
            axis=1,
        ).values
        vsx_results_dict = {}
        for row in vsx_results:
            for k, v in row.items():
                vsx_results_dict.setdefault(k, [])
                vsx_results_dict[k] += v
        vsx_types = pandas.DataFrame(vsx_results_dict)
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


class ZooniverseSubjects(ZooLookupMixin):
    def __init__(self, df=None):
        if df is not None:
            self.df = df
            return
        
        self.df = pandas.read_csv(
            os.path.join(DATA_LOCATION, 'superwasp-variable-stars-subjects.csv'),
            index_col='subject_id',
        )
    
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
    
    def display_lightcurves(self, col='lightcurve'):
        if col not in self.df:
            self.decode_locations(target=col)
            
        self.df[col].apply(
            lambda s: display(Image(url=s, width=500, height=500))
        )


class ZooniverseClassifications(object):
    ANNOTATION_PREFIX = 'annotation_'
    
    def __init__(self, df=None, drop_duplicates=False, duplicate_columns=('subject_ids', 'user_id')):
        if df is not None:
            self.df = df
            return
        
        self.df = pandas.read_csv(
            os.path.join(DATA_LOCATION, 'superwasp-variable-stars-classifications.csv'),
            index_col='classification_id',
        )
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


class FoldedLightcurves(CoordinatesMixin, ZooLookupMixin):
    def __init__(self, min_period=0, df=None):
        self.min_period = min_period
        
        if df is not None:
            self.df = df
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

    def get_siblings(self, swasp_id):
        return self.__class__(df=self.df[self.df['SWASP ID'] == swasp_id], min_period=self.min_period)


class AggregatedClassifications(CoordinatesMixin):
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

    def add_classification_labels(self):
        self.df = self.df.copy()
        self.df['Classification Label'] = self.get_classification_labels(self.df['Classification'])

    def get_classification_labels(self, series):
        return series.apply(lambda c: self.CLASSIFICATION_LABELS.get(c, None))

    def get_class(self, classification):
        return self.__class__(df=self.df[self.df['Classification'] == classification])

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
    

