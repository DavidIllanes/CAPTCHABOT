import os

exclude = ['Chimney', 'Mountain', 'Motorcycle', 'Other']
include = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Crosswalk', 'Hydrant', 'Palm', 'Traffic Light']

base_dir = '../data/'

for dir in include:
    data =

# TODO figure out class imbalance

for dirname, _, filenames in os.walk('../data/'):
    if dirname.split('/')[-1] not in exclude:
        for filename in filenames:
            print(os.path.join(dirname, filename))


