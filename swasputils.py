import os
import yaml

import pandas


def load_objects():
    objects = pandas.read_csv(
        os.path.join('superwasp-data', 'results_total.dat'),
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
    objects = objects[(objects['Period Flag'] == 0) & (objects['Period'] >= 8640000)]
    objects['SWASP ID'] = objects[objects.columns[1:3]].apply(
        lambda x: ''.join(x.astype(str)),
        axis=1
    )
    objects.drop(['SWASP', 'ID', 'Period Flag', 'Camera Number'], 'columns', inplace=True)
    return objects

def load_lookup():
    zoo_lookup = pandas.read_csv(
        os.path.join('superwasp-data', 'lookup.dat'),
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
        os.path.join('superwasp-data', 'superwasp-variable-stars-subjects.csv'),
    )[['locations', 'subject_id']]

def merge_zoo_ids(objects, lookup, fields=['SWASP ID', 'Period Number']):
    return pandas.merge(objects, lookup, left_on=fields, right_on=fields)

def merge_zoo_subjects(objects, zoo_subjects):
    return pandas.merge(objects, zoo_subjects, left_on='Zooniverse ID', right_on='subject_id').drop('subject_id', 'columns')

def decode_zoo_locations(objects):
    objects['Lightcurve'] = objects['locations'].apply(
        lambda s: yaml.load(s)['0']
    )
    return objects.drop('locations', 'columns')