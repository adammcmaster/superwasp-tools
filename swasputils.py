import os
import yaml

import pandas
from IPython.display import Image, display

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
    objects.drop(['SWASP', 'ID'], 'columns', inplace=True)
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
    return zoo_lookup

def load_zoo_subjects():
    return pandas.read_csv(
        os.path.join('superwasp-data', 'superwasp-variable-stars-subjects.csv'),
    )

def get_zoo_ids(objects, lookup, fields=['SWASP ID', 'Period Number']):
    return lookup[
        lookup[fields].apply(tuple, axis=1).isin(
            objects[fields].apply(tuple, axis=1)
        )
    ]['Zooniverse ID']

def display_zoo_lightcurves(zoo_ids, zoo_subjects):
    locations = zoo_subjects[zoo_subjects['subject_id'].isin(zoo_ids)]['locations'].apply(
        lambda s: yaml.load(s)['0']
    )
    locations.apply(lambda s: display(Image(url=s, width=500, height=500)))