#!/usr/bin/env python

from distutils.core import setup

def get_data_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
			if filename[-1] =="i":
				paths.append(os.path.join(path, filename))
    return paths

setup(name='inichord',
      version='1.0',
      description='inichord_dev',
      packages=['inichord'],
	  data_files=[('Lib/site-packages/inichord', ['inichord/config.txt']),
				  ('Lib/site-packages/inichord', get_data_files(inichord)),
     ],
     )