import os
import yaml

import pandas
from IPython.display import Image, display


DATA_LOCATION = os.path.join('..', '..', 'superwasp-data')


class ZooniverseSubjects(object):
    def __init__(self, df=None):
        if df is not None:
            self.df = df
            return
        
        self.df = pandas.read_csv(
            os.path.join(DATA_LOCATION, 'superwasp-variable-stars-subjects.csv'),
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
    
    def get_subject_set(self, set_id):
        return ZooniverseSubjects(df=self.df[self.df['subject_set_id'] == set_id])
    
    def get_workflow(self, workflow_id):
        return ZooniverseSubjects(df=self.df[self.df['workflow_id'] == workflow_id])
    
    def decode_locations(self, index=0, target='lightcurve'):
        self.df = self.df.copy()
        self.df[target] = self.df['locations'].apply(
            lambda s: yaml.full_load(s)[str(index)]
        )
    
    def display_lightcurves(self, col='lightcurve'):
        if col not in self.df:
            self.decode_locations(target=col)
            
        self.df[col].apply(
            lambda s: display(Image(url=s, width=500, height=500))
        )


class ZooniverseClassifications(object):
    def __init__(self, df=None):
        if df is not None:
            self.df = df
            return
        
        self.df = pandas.read_csv(
            os.path.join(DATA_LOCATION, 'superwasp-variable-stars-classifications.csv'),
        )
    
    @property
    def workflows(self):
        return { 
            workflow_id: self.get_workflow(workflow_id) 
            for workflow_id in set(self.df[self.df['workflow_id'].notna()]['workflow_id'])
        }
    
    def get_workflow(self, workflow_id):
        return ZooniverseClassifications(df=self.df[self.df['workflow_id'] == workflow_id])

        
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
        
    def get_siblings(self, swasp_id):
        return self.__class__(df=self.df[self.df['SWASP ID'] == swasp_id], min_period=self.min_period)


class AggregatedClassifications(object):
    def __init__(self):
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
            'Classification Count'
        ]
        self.df.drop([
            'Period',
        ], 'columns', inplace=True)


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
        
        LC_MERGE_FIELDS = ['SWASP ID', 'Period Number']
        
        self.df = folded_lightcurves.df.merge(
            self.zoo_lookup,
            left_on=LC_MERGE_FIELDS,
            right_on=LC_MERGE_FIELDS,
        ).merge(
            zooniverse_subjects.df,
            left_on='subject_id',
            right_on='subject_id',
        ).merge(
            aggregated_classifications.df,
            left_on='subject_id',
            right_on='subject_id',
        )
    
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
            swasp_id = self.df[self.df['subject_id'] == obj_id].iloc[0]['SWASP ID']
        else:
            swasp_id = obj_id
        
        return super().get_siblings(swasp_id)
    

