# sfim_lib

Functionality to be shared across SFIM users and to be made publicly available.

## Functions

Currently this package contains 4 main modules:

- `atlases` : functions for working with the Schafer atlas
- `fc_introspection` : functions and variables specifically for working with the FC Introspection project, including directory structure, loading data, SYNCQ information.
- `io` : functions for loading in data
- `plotting` : functions for plotting data. Specifically, contains functions for plotting FC matrix data.

Other modules may be added as new functionality is developed.

## Installation

Currently, the easiest way to utilize these functions is to use the pip developers mode. To utilize this functionality, clone this repo to wherever you would like to use it (either locally or on Biowulf). From there, navigate into to that cloned directory and pip install:

`pip install -e .`

After running this command, you can import the functions as you would for any Python package. By installing using the developer's editable mode, if you change any of the functions in the package and save them as you normally would, all you have to do is re-import the package in Python for them to be applied.

For example, to import functions and variables from the `fc_introspection` module, one would use the command:
`from sfim_lib.fc_introspection.initialize_variables import *` - this command will import all necessary variables and directory structures for this project, in addition to the function for loading subject data.

## Dependencies

This package has been tested on Python version 3.12.2. Previous verions of Python (and the packages below) may still work, they just have not been tested.

The primary package dependencies needed to run this code are:

- pandas 2.1.4
- numpy 1.26.4
- os

The plotting functionality additionally requires:

- bokeh 3.3.4
- nilearn 0.10.3
- holoviews 1.18.3
- hvplot 0.9.2
- matplotlib 3.8.0
- networkx 3.1
- nxviz 0.7.4
  - this package seems to be particularly problematic with some deprecation, but there is fucntionality in this submodule that requires this package. Eventually, this will be updated so this issue is fixed
- scikit-learn 1.4.1.post1

## Contributing

If you develop a function that you think will be useful for the rest of the lab, the best way to integrate them is to create a new branch and submit a pull request. Please see [instructions on the lab documentation](https://github.com/nimh-sfim/lab-docs/blob/main/editing.md) for detailed instructions on how to do this.
