# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['libtpu']

package_data = \
{'': ['*']}

setup_kwargs = {
    'name': 'libtpu',
    'version': '0.1.0',
    'description': 'A reference libtpu.so implementation for Python and JAX',
    'long_description': '',
    'author': 'Shawn Presser',
    'author_email': 'None',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'python_requires': '>=3.6,<4.0',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
