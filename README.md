# pythreadex
Pythreadex is a simple Python utility to fetch long-term station data for the United States from ThreadEx, courtesy of ACIS (https://www.rcc-acis.org/). This utility includes a few plotting and climatology functions, as well as converting data to CSV format.

## Installation


### Pip

Installation is available via pip:

```sh
pip install pythreadex
```

### From source

pythreadex can also be installed from source by cloning the GitHub repository:

```sh
git clone https://github.com/tomerburg/pythreadex
cd pythreadex
python setup.py install
```

## Dependencies
- matplotlib >= 2.2.2
- numpy >= 1.14.3
- scipy >= 1.1.0
- pandas >= 1.3.0

## Sample Usage
For full documentation and examples, please refer to [Tropycal Documentation](https://tropycal.github.io/tropycal/).

As of v0.3, the documentation is up-to-date following a bug that started with v0.2.5 where the documentation was not updated with each release.

## Sample Usage
This sample code shows how to search through a dataset, retrieve station data, make a plot and convert the data to CSV format:

```python
from pythreadex import Dataset
import matplotlib.pyplot as plt
            
# Create an instance of Dataset
dataset = Dataset()

# Retrieve all stations in New Jersey
print(stations.search_by_state('NJ'))

# Search for Newark, NJ's station ID
station_id = dataset.search_by_name('Newark, NJ')

# Create an instance of a Station object with this station ID
station = dataset.get_station(station_id)

# Make a plot of 2023 maximum temperatures, from January to May
station.plot_temp_time_series('max', year=2023, date_range=('1/1','5/31'))

# Convert data to a CSV file
station.to_csv('newark.csv')
```
