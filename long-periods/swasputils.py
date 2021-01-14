import os
import yaml

import pandas
from IPython.display import Image, display


DATA_LOCATION = os.path.join('..', '..', 'superwasp-data')

def load_objects(min_period=8640000):
    objects = pandas.read_csv(
        os.path.join(DATA_LOCATION, 'results_total.dat'),
        delim_whitespace=True,
        header=None,
    )
    objects.columns = [
        'Camera Number',
        'SWASP',
        'ID',
        'Period Number',
        'Period',
        'Sigma',
        'Chi Squared',
        'Period Flag'
    ]
    objects = objects[(objects['Period Flag'] == 0) & (objects['Period'] >= min_period)]
    objects.drop(['Period Flag', 'Camera Number'], 'columns', inplace=True)
    objects['SWASP ID'] = objects['SWASP'] + objects['ID']
    return objects

def load_lookup():
    zoo_lookup = pandas.read_csv(
        os.path.join(DATA_LOCATION, 'lookup.dat'),
        delim_whitespace=True,
        header=None,
    )
    zoo_lookup.columns = [
        'Zooniverse ID',
        'SWASP ID',
        'Period',
        'Period Number',
    ]
    zoo_lookup.drop('Period', 'columns', inplace=True)
    return zoo_lookup

def load_zoo_subjects():
    return pandas.read_csv(
        os.path.join(DATA_LOCATION, 'superwasp-variable-stars-subjects.csv'),
    )[['locations', 'subject_id']]

def load_classifications():
    classifications = pandas.read_csv(
        os.path.join(DATA_LOCATION, 'class_top.csv'),
        delim_whitespace=True,
        header=None,
    )
    classifications.columns = [
        'Zooniverse ID',
        'SWASP ID',
        'Period Number',
        'Period',
        'Classification',
        'Period Uncertainty',
        'Classification Count'
    ]
    classifications.drop([
        'SWASP ID',
        'Period Number',
        'Period',
    ], 'columns', inplace=True)
    return classifications

def load_manual_classifications():
    return pandas.read_csv(
        os.path.join(DATA_LOCATION, 'superwasp-long-periods-classifications.csv'),
    )[['subject_ids', 'annotations']]

def merge_zoo_ids(objects, lookup):
    MERGE_FIELDS = ['SWASP ID', 'Period Number']
    return pandas.merge(objects, lookup, left_on=MERGE_FIELDS, right_on=MERGE_FIELDS)

def merge_zoo_subjects(objects, zoo_subjects):
    return pandas.merge(
        objects,
        zoo_subjects,
        left_on='Zooniverse ID',
        right_on='subject_id',
    ).drop('subject_id', 'columns')

def merge_classifications(objects, classifications):
    return pandas.merge(
        objects,
        classifications,
        left_on='Zooniverse ID',
        right_on='Zooniverse ID',
        how='left',
    )

def merge_manual_classifications(objects, classifications):
    return pandas.merge(
        objects,
        classifications,
        left_on='Zooniverse ID',
        right_on='subject_ids',
        how='left',
    ).drop('subject_ids', 'columns')

def decode_zoo_locations(objects):
    objects['Lightcurve'] = objects['locations'].apply(
        lambda s: yaml.load(s)['0']
    )
    return objects.drop('locations', 'columns')

def decode_manual_annotations(objects):
    def decode(s):
        try:
            return yaml.load(s)[0]['value']
        except (AttributeError, IndexError, KeyError):
            return None
    objects['Manual Classification'] = objects['annotations'].apply(decode)
    return objects.drop('annotations', 'columns')