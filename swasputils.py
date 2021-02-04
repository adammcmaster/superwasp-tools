import os

import pandas
import ujson

from IPython.display import Image, display


DATA_LOCATION = os.path.join('..', '..', 'superwasp-data')


class ZooniverseSubjects(object):
    def __init__(self, df=None):
        if df is not None:
            self.df = df
            return
        
        self.df = pandas.read_csv(
            os.path.join(DATA_LOCATION, 'superwasp-variable-stars-subjects.csv'),
            index_col='subject_id'
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


class FoldedLightcurves(object):
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

        def decode_coords(self):
            if 'ra' in self.df and 'dec' in self.df:
                return

            #self.df
            coords = superwasp_id.replace('1SWASP', '')
            coords_quoted = urllib.parse.quote(coords)
            ra = urllib.parse.quote('{}:{}:{}'.format(
                coords[1:3],
                coords[3:5],
                coords[5:10]
            ))
            dec = urllib.parse.quote('{}:{}:{}'.format(
                coords[10:13],
                coords[13:15],
                coords[15:]
            ))
        
    def get_siblings(self, swasp_id):
        return self.__class__(df=self.df[self.df['SWASP ID'] == swasp_id], min_period=self.min_period)


class AggregatedClassifications(object):
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
        
        self.df = zooniverse_subjects.df.reset_index().merge(
            self.zoo_lookup,
            how='left',
        ).merge(
            aggregated_classifications.df.reset_index(),
            how='left',
        ).merge(
            folded_lightcurves.df.reset_index(),
            how='left',
        ).set_index('subject_id')
    
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
        zoo_lookup.drop('Period', 'columns', inplace=True)
        return zoo_lookup
    
    def get_siblings(self, obj_id):
        if type(obj_id) == int:
            swasp_id = self.df[self.df.index == obj_id].iloc[0]['SWASP ID']
        else:
            swasp_id = obj_id
        
        return super().get_siblings(swasp_id)
    

