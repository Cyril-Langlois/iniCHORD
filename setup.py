#!/usr/bin/env python

from distutils.core import setup

setup(name='inichord',
      version='1.0',
      description='inichord_dev',
      packages=['inichord'],
	  data_files=[('Lib/site-packages/inichord', ['inichord/config.txt']),
     ],
     )