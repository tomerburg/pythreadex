import io
import json
import urllib
import calendar
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class Dataset:
    
    def __init__(self):

        # Fetch all station data from ThreadEx
        url = 'https://threadex.rcc-acis.org'
        f = urllib.request.urlopen(url)
        content = (f.read()).decode('utf-8')
        f.close()
        self.meta = {}
        for line in content.split('\n'):
            if '<li data-value="' in line:
                station_id = (line.split('<li data-value="')[1]).split('"')[0]
                station_info = (line.split('<div>')[1]).split('</div>')[0]
                if '-' not in station_info:
                    continue
                station_state = ((station_info.split('-')[0]).rstrip()).lstrip()
                station_name = '-'.join(station_info.split('-')[1:])
                station_name = ((station_name).rstrip()).lstrip()

                self.meta[station_id] = {
                    'station_id': station_id,
                    'state': station_state,
                    'name': station_name
                }
    
    def search_by_state(self, state):
        r"""
        Returns all station names and IDs for this state.
        
        Parameters
        ----------
        state : str
            State abbreviation (e.g., "NY") to search stations for.
        
        Returns
        -------
        dict
            Dictionary mapping all station IDs (key) to their names (value) for this state.
        """
        
        stations = {}
        for key in self.meta:
            if self.meta[key]['state'] == state:
                stations[self.meta[key]['station_id']] = f'{self.meta[key]["name"]}, {self.meta[key]["state"]}'
        return stations
    
    def search_by_name(self, name):
        r"""
        Returns a station ID from the requested station name. Query is case insensitive.
        
        Parameters
        ----------
        name : str
            City name and state abbreviation, separated by a comma and space (e.g., "Burlington, VT").
        
        Returns
        -------
        str
            Station ID corresponding to ThreadEx.
        """
        
        for key in self.meta:
            if name.lower() == f'{self.meta[key]["name"]}, {self.meta[key]["state"]}'.lower():
                return key
        raise ValueError("Could not find a matching station.")
    
    def get_station(self, station_id, metric=False):
        r"""
        Returns a Station object for the requested station ID.
        
        Parameters
        ----------
        station_id : str
            Station ID string. Can be 3-letter airport identifier (e.g., "JFK"), or the full ThreadEx station identifier.
        metric : bool, optional
            If True, uses the metric system for all units. Default is False.
        
        Returns
        -------
        pythreadex.Station
            An instance of a Station object containing data and plotting methods for this station.
        
        Notes
        -----
        The station ID returned by `Dataset.search_by_name()` can be plugged in for `station_id`. Below is an example for Newark, NJ:
        
        .. code-block:: python
        
            from pyhreadex import Dataset
            
            # Create an instance of Dataset
            dataset = Dataset()
            
            # Search for Newark, NJ's station ID
            station_id = dataset.search_by_name('Newark, NJ')
            
            # Create an instance of a Station object with this station ID
            station = dataset.get_station(station_id)
            
        """
        
        # Get station ID
        if station_id[-3:] != 'thr':
            station_id += 'thr'
        if station_id not in self.meta:
            raise ValueError("Requested station does not exist.")
        
        # Download data from ThreadEx
        url = f'https://data.rcc-acis.org/StnData?sid={station_id}%209&sdate=por&edate=por&elems=1,2,4,10,11&output=csv'
        f = urllib.request.urlopen(url)
        data = (f.read()).decode('utf-8')
        data = 'date,max_temp,min_temp,precip,snow,snow_depth\n' + '\n'.join(data.split('\n')[1:])
        f.close()
        
        return Station(io.StringIO(data), self.meta[station_id], metric)

class Station:
    
    def __repr__(self):

        # Label object
        summary = ["<pythreadex.Station>"]
        
        # Format keys
        unit_label = 'F'
        if self.metric:
            unit_label = 'C'
        degree_sign = u'\N{DEGREE SIGN}'
        summary_keys = {
            'Highest Maximum Temp': f'{self.attrs["max_maxtemp"]["value"]}{degree_sign}{unit_label} ({self.attrs["max_maxtemp"]["date"].strftime("%B %d, %Y")})',
            'Lowest Maximum Temp': f'{self.attrs["min_maxtemp"]["value"]}{degree_sign}{unit_label} ({self.attrs["min_maxtemp"]["date"].strftime("%B %d, %Y")})',
            'Highest Minimum Temp': f'{self.attrs["max_mintemp"]["value"]}{degree_sign}{unit_label} ({self.attrs["max_mintemp"]["date"].strftime("%B %d, %Y")})',
            'Lowest Minimum Temp': f'{self.attrs["min_mintemp"]["value"]}{degree_sign}{unit_label} ({self.attrs["min_mintemp"]["date"].strftime("%B %d, %Y")})',
        }

        # Add station attributes
        summary.append("Station Highlights:")
        add_space = np.max([len(key) for key in summary_keys.keys()]) + 3
        for key in summary_keys.keys():
            key_name = key + ":"
            summary.append(
                f'{" "*4}{key_name:<{add_space}}{summary_keys[key]}')

        # Format keys
        summary_keys = {
            'Station Name': self.attrs['name'],
            'Station State': self.attrs['state'],
            'Start Date': self.attrs['start_date'].strftime('%B %d, %Y'),
            'End Date': self.attrs['end_date'].strftime('%B %d, %Y'),
        }
        if 'station_id' in self.attrs:
            summary_keys['Station ID'] = self.attrs['station_id']

        # Add station attributes
        summary.append("\nStation Attributes:")
        add_space = np.max([len(key) for key in summary_keys.keys()]) + 3
        for key in summary_keys.keys():
            key_name = key + ":"
            summary.append(
                f'{" "*4}{key_name:<{add_space}}{summary_keys[key]}')

        return "\n".join(summary)
    
    def __init__(self, filename, attrs, metric=False):
        r"""
        Initializes an instance of Station.
        
        Parameters
        ----------
        filename : str
            File name of CSV file to read containing ThreadEx data. If retrieved using ``dataset.get_station()``, this is automatically provided.
        attrs : dict
            Dictionary containing metadata for this station. If retrieved using ``dataset.get_station()``, this is automatically provided.
        metric : bool, optional
            If True, uses the metric system for all units. Default is False.
        
        Returns
        -------
        pythreadex.Station
            An instance of a Station object containing data and plotting methods for this station.
        
        Notes
        -----
        If reading from a locally stored CSV file, attrs must be provided by the user. This is a dictionary that must contain the keys "name" and "state", for example as follows:
        
        .. code-block:: python
        
            attrs = {
                'name': 'New York',
                'state': 'NY'
            }
            station = Station('my_csv_file.csv', attrs)
        
        """
        
        # Convert data to Pandas DataFrame
        self.data = pd.read_csv(filename, na_values='M')
        self.data_original = self.data.copy()
        
        # Set basic QC thresholds
        qc_thresh = {
            'max_maxt': 140,
            'min_maxt': -70,
            'max_mint': 95,
            'min_mint': -100
        }
        
        # Convert to metric if requested
        def multiply_numbers(value, factor):
            try:
                return str(round(float(value) * factor,1))
            except ValueError:
                return value
        self.metric = metric
        if metric:
            self.data['max_temp'] = round((self.data['max_temp'] - 32) * (5/9),1)
            self.data['min_temp'] = round((self.data['min_temp'] - 32) * (5/9),1)
            self.data['precip'] = self.data['precip'].apply(lambda x: multiply_numbers(x, 25.4))
            self.data['snow'] = self.data['snow'].apply(lambda x: multiply_numbers(x, 2.54))
            self.data['snow_depth'] = self.data['snow_depth'].apply(lambda x: multiply_numbers(x, 2.54))
            for key in qc_thresh:
                qc_thresh[key] = (qc_thresh[key] - 32) * (5/9)
        
        # Fetch more metadata, if possible
        self.attrs = attrs
        self.attrs['metric'] = metric
        self.attrs['start_date'] = dt.datetime.strptime(self.data['date'].values[0],'%Y-%m-%d')
        self.attrs['end_date'] = dt.datetime.strptime(self.data['date'].values[-1],'%Y-%m-%d')
        if 'name' not in self.attrs or 'state' not in self.attrs:
            try:
                url = f'https://data.rcc-acis.org/StnMeta?sids={self.attrs["station_id"]}%209&meta=name,state,sid_dates,ll,elev,county'
                f = urllib.request.urlopen(url)
                data = json.loads((f.read()).decode('utf-8'))
                f.close()
            
                if 'name' not in self.attrs:
                    self.attrs['name'] = data['meta'][0]['name']
                if 'state' not in self.attrs:
                    self.attrs['state'] = data['meta'][0]['state']
            except:
                pass
        
        # Store filtered station data
        self.filtered_data = {}
        
        # Max filtering
        max_conditions = (~np.isnan(self.data['max_temp'])) & (self.data['max_temp'] < qc_thresh['max_maxt']) & (self.data['max_temp'] > qc_thresh['min_maxt'])
        self.filtered_data['max'] = self.data.loc[max_conditions].copy()
        self.filtered_data['max'].rename(columns={'max_temp':'max','min_temp':'min'}, inplace=True)

        # Min filtering
        min_conditions = (~np.isnan(self.data['min_temp'])) & (self.data['min_temp'] < qc_thresh['max_mint']) & (self.data['min_temp'] > qc_thresh['min_mint'])
        self.filtered_data['min'] = self.data.loc[min_conditions].copy()
        self.filtered_data['min'].rename(columns={'max_temp':'max','min_temp':'min'}, inplace=True)

        # Combined filtering
        both_conditions = min_conditions & (self.filtered_data['max']['min'] < self.filtered_data['max']['max'])
        self.filtered_data['both'] = self.filtered_data['max'].loc[both_conditions].copy()
        
        # Add station extrema to attributes
        value = np.nanmax(self.filtered_data['max']['max'])
        date = self.data.loc[self.data['max_temp'] == value].tail(1)['date'].values[0]
        self.attrs['max_maxtemp'] = {
            'value': round(value, 1) if metric else int(value),
            'date': dt.datetime.strptime(date,'%Y-%m-%d')
        }
        value = np.nanmin(self.filtered_data['max']['max'])
        date = self.data.loc[self.data['max_temp'] == value].tail(1)['date'].values[0]
        self.attrs['min_maxtemp'] = {
            'value': round(value, 1) if metric else int(value),
            'date': dt.datetime.strptime(date,'%Y-%m-%d')
        }
        value = np.nanmax(self.filtered_data['min']['min'])
        date = self.data.loc[self.data['min_temp'] == value].tail(1)['date'].values[0]
        self.attrs['max_mintemp'] = {
            'value': round(value, 1) if metric else int(value),
            'date': dt.datetime.strptime(date,'%Y-%m-%d')
        }
        value = np.nanmin(self.filtered_data['min']['min'])
        date = self.data.loc[self.data['min_temp'] == value].tail(1)['date'].values[0]
        self.attrs['min_mintemp'] = {
            'value': round(value, 1) if metric else int(value),
            'date': dt.datetime.strptime(date,'%Y-%m-%d')
        }

    def get_data(self):
        r"""
        Returns Pandas DataFrame containing raw data for this station.
        
        Returns
        -------
        Pandas.DataFrame
            Data associated with this station.
        """
        
        return self.data
    
    def to_csv(self, filename):
        r"""
        Save data as a comma delimited file (CSV).
        
        Parameters
        ----------
        filename : str
            Filename to save CSV file to.
        """
        
        self.data_original.set_index(self.data.columns[0]).to_csv(filename)
    
    def get_daily_climatology(self, date):
        r"""
        Returns all data for the given calendar date.
        
        Parameters
        ----------
        date : datetime.datetime
            Datetime object containing the requested date. The year is ignored for this purpose.
        
        Returns
        -------
        Pandas.DataFrame
            Filtered data for this calendar date.
        """
        
        return self.data[self.data['date'].str.contains(date.strftime('-%m-%d$'))]
        
    def calc_temperature_climatology(self, modes, climatology=None, return_df=False):
        r"""
        Calculates a smoothed climatology for a given climatological range.
        
        Parameters
        ----------
        modes : list or str
            Mode or list of modes to calculate climatology for. Accepted values are "max", "min" and "average".
        climatology : tuple, optional
            Tuple with 2 elements denoting the start and end years to calculate climatology for (e.g., `(1991, 2020)`). If None (default), this is automatically determined.
        return_df : bool, optional
            By default (False), a dict is returned. If True, a formatted pandas DataFrame is returned. 
        
        Returns
        -------
        dict or pandas.DataFrame
            Dictionary or pandas DataFrame containing the climatology average and percentile ranks for the climatology period, and extrema for the entire dataset.
        
        Notes
        -----
        The climatology is computed by a rolling 24-day centered window weighted towards the center, with additional Gaussian filtering applied for temporal smoothing.
        """
        
        # Retrieve filtered data
        if not isinstance(modes,list):
            modes = [modes]
        if modes == ['max'] or modes == ['min']:
            data = self.filtered_data[modes[0]].copy()
        else:
            data = self.filtered_data['both'].copy()
        
        # Convert data to a dictionary
        dates = [pd.to_datetime(i) for i in data['date']]
        tmax = data['max'].values
        tmin = data['min'].values
        data = {}
        for i,(i_tmax,i_tmin,i_date) in enumerate(zip(tmax,tmin,dates)):
            if i_date.strftime('%m%d') == '0229': continue

            i_year = i_date.year
            if i_year not in data.keys():
                data[i_year] = {'days':[],'dates':[],'max':[],'min':[],'average':[]}

            data[i_year]['days'].append((i_date.replace(year=2015)-dt.datetime(2015,1,1)).total_seconds() / 86400)
            data[i_year]['dates'].append(i_date)
            data[i_year]['max'].append(i_tmax)
            data[i_year]['min'].append(i_tmin)
            data[i_year]['average'].append((i_tmax+i_tmin)/2.0)
        
        # Get year range for climatology
        years = np.unique([int(i.year) for i in dates])
        if climatology is None:
            nearest_end_year = round_to_ten(years[-2])-10
            climatology = (nearest_end_year-29,nearest_end_year)
            if nearest_end_year-29 < years[0]:
                nearest_start_year = round_to_ten(years[0])+1
                if nearest_start_year-10 in years: nearest_start_year = nearest_start_year - 10
                climatology = (nearest_start_year,nearest_start_year+29)
        if climatology[1]-climatology[0] < 29:
            raise ValueError("Error: Climatology must have at least 30 years.")

        # Calculate climatology
        def init_climo_structure():
            return {
                'days': list(range(365)), 'average': [], 'min': [], 'p10': [],
                'p25': [], 'p75': [], 'p90': [], 'max': [], 'climatology': climatology,
                'extrema': (years[0], years[-1])
            }
        climo = {k: init_climo_structure() for k in modes}

        # Create primary climatology with a weighted rolling window
        iterate_fac = [1]*6 + [2]*3 + [3]*2 + [4] + [5] + [4] + [3]*2 + [2]*3 + [1]*6
        for i in range(365):
            temp_data = {}
            for mode in modes:
                temp_data[mode] = []

            # Iterate over all years in climatology
            iterate_i = [i+k for k in range(-12, 13)]
            iterate_i = [(k-365 if k >= 365 else k) for k in iterate_i]
            for i_year in range(climatology[0], climatology[1] + 1):
                
                # Filtering
                if i_year not in data:
                    continue
                i_year_data = data[i_year]
                if i not in i_year_data['days']:
                    continue

                # Append data
                for iter_i, iter_fac in zip(iterate_i, iterate_fac):
                    if iter_i in i_year_data['days']:
                        idx_iter = i_year_data['days'].index(iter_i)

                        values = {}
                        for mode in modes:
                            values[mode] = i_year_data[mode][idx_iter]

                        for key, value in values.items():
                            temp_data[key].extend([value] * iter_fac)

            # Calculate percentile ranks and averages
            for rank in [10, 25, 75, 90]:
                for mode in modes:
                    climo[mode][f'p{rank}'].append(np.nanpercentile(temp_data[mode], rank))
            for mode in modes:
                climo[mode]['average'].append(np.nanmean(temp_data[mode]))

        # Apply gaussian filter to smooth data
        for key in ['p10', 'p25', 'average', 'p75', 'p90']:
            for mode in modes:
                climo[mode][key] = gaussian_filter(climo[mode][key], sigma=1.0)
        
        # Calculate dataset extrema
        keys = list(data.keys())
        for i in range(365):
            temp_data = {}
            for mode in modes:
                temp_data[mode] = []

            # Iterate over all years
            for i_year in keys:
                
                # Filtering
                if i_year not in data:
                    continue
                i_year_data = data[i_year]
                if i not in i_year_data['days']:
                    continue
                idx = i_year_data['days'].index(i)
                if i_year == keys[-1] and i_year_data['dates'][idx] > dates[-1]:
                    continue

                # Append data
                values = {}
                for mode in modes:
                    values[mode] = i_year_data[mode][idx]
                for key, value in values.items():
                    temp_data[key].append(value)

            for mode in modes:
                climo[mode]['min'].append(np.nanmin(temp_data[mode]))
                climo[mode]['max'].append(np.nanmax(temp_data[mode]))
        
        if not return_df:
            return climo
        df = {}
    
    def plot_temp_time_series(self, mode, year=None, date_range=None, climatology=None,
                              plot_rectangle=False, plot_anomaly=False):
        r"""
        Plots a time series of station temperatures.
        
        Parameters
        ----------
        mode : str
            Plotting mode. Can be "max", "min", "average", or "both".
        year : int, optional
            Year to plot. If None (default), no year is plotted.
        date_range : list or tuple, optional
            List or tuple with 2 elements denoting the start and end dates (e.g., ``['7/1', '11/30']``). If None (default), the full year is plotted.
        climatology : tuple, optional
            Tuple with 2 elements denoting the start and end years to calculate climatology for. If None (default), this is automatically determined.
        plot_rectangle : bool, optional
            If True, rectangles are plotted when ``mode='both'``. Default is False.
        plot_anomaly : bool, optional
            If True, temperature anomaly is plotted instead of quantiles for applicable modes. Default is False.
        
        Returns
        -------
        matplotlib.pyplot.axes
            Axes instance containing the plot.
        
        Notes
        -----
        This plotting function ignores February 29th.
        """
        
        # Apply basic quality control to data
        if mode not in ['max', 'min', 'average', 'both']:
            raise ValueError('Acceptable values for "mode" are "max", "min", "average" or "both".')
        if mode == 'average':
            data = self.filtered_data['both'].copy()
        else:
            data = self.filtered_data[mode].copy()
        if date_range is None:
            date_range = ['1/1', '12/31']

        # Convert data to a dictionary
        dates = [pd.to_datetime(i) for i in data['date']]
        tmax = data['max'].values
        tmin = data['min'].values
        data = {}
        for i,(i_tmax,i_tmin,i_date) in enumerate(zip(tmax,tmin,dates)):
            if i_date.strftime('%m%d') == '0229': continue

            i_year = i_date.year
            if i_year not in data.keys():
                data[i_year] = {'days':[],'dates':[],'max':[],'min':[],'average':[]}

            data[i_year]['days'].append((i_date.replace(year=2015)-dt.datetime(2015,1,1)).total_seconds() / 86400)
            data[i_year]['dates'].append(i_date)
            data[i_year]['max'].append(i_tmax)
            data[i_year]['min'].append(i_tmin)
            data[i_year]['average'].append((i_tmax+i_tmin)/2.0)

        # Get year range for climatology
        years = np.unique([int(i.year) for i in dates])
        if climatology is None:
            if year is None:
                nearest_end_year = round_to_ten(years[-2])-10
                climatology = (nearest_end_year-29,nearest_end_year)
            else:
                nearest_end_year = round_to_ten(year)-10
                climatology = (nearest_end_year-29,nearest_end_year)
            if nearest_end_year-29 < years[0]:
                nearest_start_year = round_to_ten(years[0])+1
                if nearest_start_year-10 in years: nearest_start_year = nearest_start_year - 10
                climatology = (nearest_start_year,nearest_start_year+29)
        if climatology[1]-climatology[0] < 29:
            raise ValueError("Error: Climatology must have at least 30 years.")
        
        # Calculate climatology
        submit_mode = mode if mode != 'both' else ['max', 'min']
        climo = self.calc_temperature_climatology(submit_mode, climatology)

        # -------------------------------------------------------------------------------------
        
        # Create figure
        fig,ax = plt.subplots(figsize=(9,6),dpi=200)

        # Configure x-axis
        start_lim = dt.datetime.strptime(f'2015/{date_range[0]}','%Y/%m/%d')
        end_lim = dt.datetime.strptime(f'2015/{date_range[1]}','%Y/%m/%d')
        diff_lim = (end_lim-start_lim).total_seconds()/86400
        if diff_lim > 120:
            julian_data = months_in_julian(2015)
            xticks = julian_data['start']
            for i,(istart,iend) in enumerate(zip(xticks[::2],xticks[1:][::2])):
                ax.axvspan(istart,iend,color='#e4e4e4',alpha=0.5,zorder=0)
                ax.axvline(istart,linestyle='-',color='k',zorder=20,alpha=0.3,linewidth=0.5)
                ax.axvline(iend,linestyle='-',color='k',zorder=20,alpha=0.3,linewidth=0.5)
            ax.set_xticks(julian_data['midpoint'])
            ax.set_xticklabels(julian_data['name'])
        else:
            xticks = []
            xticklabels = []
            iter_lim = start_lim + dt.timedelta(hours=0)
            while iter_lim <= end_lim + dt.timedelta(hours=24*7):
                xticks.append(convert_to_julian(iter_lim))
                xticklabels.append(iter_lim.strftime('%b %d'))
                iter_lim += dt.timedelta(hours=24*7)
            for i,(istart,iend) in enumerate(zip(xticks[::2],xticks[1:][::2])):
                ax.axvspan(istart,iend,color='#e4e4e4',alpha=0.5,zorder=0)
                ax.axvline(istart,linestyle='-',color='k',zorder=20,alpha=0.3,linewidth=0.5)
                ax.axvline(iend,linestyle='-',color='k',zorder=20,alpha=0.3,linewidth=0.5)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
        unit_label = 'F'
        if self.metric:
            unit_label = 'C'
        degree_sign = u'\N{DEGREE SIGN}'
        ax.set_ylabel(f'Temperature ({degree_sign}{unit_label})')

        # Plot climatological data
        if mode == 'both':
            ax.fill_between(climo['min']['days'],climo['min']['min'],climo['min']['average'],
                            color='#C8E4F9',zorder=3,alpha=0.5)
            ax.fill_between(climo['min']['days'],climo['max']['average'],climo['max']['max'],
                            color='#F9C8C8',zorder=3,alpha=0.5)
            ax.fill_between(climo['min']['days'],climo['min']['average'],climo['max']['average'],
                            color='#F4EB7F',zorder=4,alpha=0.5,label='Average Temp')
            ax.plot(climo['min']['days'],climo['min']['min'],
                    'b',linewidth=0.5,alpha=0.3,linestyle='solid',zorder=6,label=f'Record Minimum')
            ax.plot(climo['min']['days'],climo['max']['max'],
                    'r',linewidth=0.5,alpha=0.3,linestyle='solid',zorder=6,label=f'Record Maximum')
            ax.plot(climo['min']['days'],climo['min']['average'],'#DDD03C',
                    linewidth=0.8,zorder=5,linestyle='solid')
            ax.plot(climo['min']['days'],climo['max']['average'],'#DDD03C',
                    linewidth=0.8,zorder=5,linestyle='solid')
        else:
            color_store = {
                'max': {'10-90': '#F9C8C8', '25-75': '#F48C8C', 'mean': '#B10000'},
                'min': {'10-90': '#C8E4F9', '25-75': '#8CC8F4', 'mean': 'b'},
                'average': {'10-90': '#E0DAE8', '25-75': '#C3BDCB', 'mean': '#6E6B71'},
            }
            kwargs = {'alpha':0.3} if plot_anomaly else {}
            ax.fill_between(climo[mode]['days'],climo[mode]['p10'],climo[mode]['p90'],
                            zorder=3,color=color_store.get(mode)['10-90'],label='10 - 90th Percentile',**kwargs)
            ax.fill_between(climo[mode]['days'],climo[mode]['p25'],climo[mode]['p75'],
                            zorder=3,color=color_store.get(mode)['25-75'],label='25 - 75th Percentile',**kwargs)
            ax.plot(climo[mode]['days'],climo[mode]['min'],'b',linewidth=0.7,zorder=4,
                    linestyle='solid',label=f'Record Minimum')
            ax.plot(climo[mode]['days'],climo[mode]['max'],'r',linewidth=0.7,zorder=4,
                    linestyle='solid',label=f'Record Maximum')
            ax.plot(climo[mode]['days'],climo[mode]['average'],color_store.get(mode)['mean'],
                    linewidth=2.0,zorder=4,label=f'Average')

        # Plot requested season
        keys = [k for k in data.keys()]
        if year is not None:
            if year == keys[-1]:
                idx = data[year]['dates'].index(dates[-1])
                plot_x = data[year]['days'][:idx+1]
                plot_y_tmin = data[year]['min'][:idx+1]
                plot_y_tmax = data[year]['max'][:idx+1]
                plot_y_tavg = data[year]['average'][:idx+1]
                if mode != 'both': plot_avg = climo[mode]['average'][:idx+1]
            else:
                plot_x = data[year]['days']
                plot_y_tmin = data[year]['min']
                plot_y_tmax = data[year]['max']
                plot_y_tavg = data[year]['average']
                if mode != 'both': plot_avg = climo[mode]['average']
            lw = 1.2 if plot_anomaly else 2.0
            if mode == 'min':
                plot_y_val = plot_y_tmin
                ax.plot(plot_x,plot_y_tmin,linewidth=lw,color='k',zorder=9,label=f'Year: {year}')
            elif mode == 'max':
                plot_y_val = plot_y_tmax
                ax.plot(plot_x,plot_y_tmax,linewidth=lw,color='k',zorder=9,label=f'Year: {year}')
            elif mode == 'average':
                plot_y_val = plot_y_tavg
                ax.plot(plot_x,plot_y_tavg,linewidth=lw,color='k',zorder=9,label=f'Year: {year}')
            elif plot_rectangle:
                for i,(i_min,i_max) in enumerate(zip(plot_y_tmin,plot_y_tmax)):
                    ax = add_rectangle(ax,i,i_min,i_max,color='#178FCB',width=0.9,linewidth=0,alpha=0.8)
            else:
                ax.plot(plot_x,plot_y_tmin,linewidth=1.2,color='b',zorder=9,label=f'{year} Min Temp')
                ax.plot(plot_x,plot_y_tmax,linewidth=1.2,color='r',zorder=9,label=f'{year} Max Temp')

            # Plot anomaly, if requested and applicable
            if plot_anomaly and mode in ['min', 'max', 'average']:
                new_x = np.arange(plot_x[0],plot_x[-1]+0.1,0.1)
                plot_avg = np.interp(new_x, plot_x, plot_avg)
                plot_y_val = np.interp(new_x, plot_x, plot_y_val)
                plot_x = new_x
                ax.fill_between(plot_x,plot_avg,plot_y_val,where=plot_y_val>=plot_avg,
                                color='r',alpha=0.4,zorder=3,label='Above Average')
                ax.fill_between(plot_x,plot_avg,plot_y_val,where=plot_y_val<=plot_avg,
                                color='b',alpha=0.4,zorder=3,label='Below Average')

            # Highlight record lows & highs
            if mode in ['min', 'max', 'average']:
                iter_temp_min = data[year][mode]
                iter_temp_max = data[year][mode]
            else:
                iter_temp_min = data[year]['min']
                iter_temp_max = data[year]['max']
            iter_climo = {}
            if mode in ['min', 'max', 'average']:
                iter_climo['min'] = climo[mode]['min']
                iter_climo['max'] = climo[mode]['max']
            else:
                iter_climo['min'] = climo['min']['min']
                iter_climo['max'] = climo['max']['max']
            for i in range(365):
                if i not in data[year]['days']: continue
                i_temp = iter_temp_min[data[year]['days'].index(i)]
                i_record = iter_climo['min'][i]
                if i_temp <= i_record: ax.plot(i,i_temp,'o',zorder=11,mfc='#8CC8F4',ms=8,mec='k')
                i_temp = iter_temp_max[data[year]['days'].index(i)]
                i_record = iter_climo['max'][i]
                if i_temp >= i_record: ax.plot(i,i_temp,'o',zorder=11,mfc='r',ms=8,mec='k')

        # Set limit of days
        start_count = int((start_lim - dt.datetime(2015,1,1)).total_seconds()/86400)
        end_count = int((end_lim - dt.datetime(2015,1,1)).total_seconds()/86400)
        climo_mode = 'max' if mode == 'both' else mode
        ax.set_xlim(climo[climo_mode]['days'][start_count],climo[climo_mode]['days'][end_count])

        # Determine y-lim based on subset data
        check_store = {
            'max':{'max':'max','min':'max'},
            'min':{'max':'min','min':'min'},
            'average':{'max':'average','min':'average'},
            'both':{'max':'max','min':'min'}
        }
        max_val = np.nanmax(climo[check_store.get(mode)['max']]['max'][start_count:end_count+1])
        min_val = np.nanmin(climo[check_store.get(mode)['min']]['min'][start_count:end_count+1])
        data_range = max_val - min_val
        min_val = min_val - (data_range*0.1)
        max_val = max_val + (data_range*0.1)
        ax.set_ylim(min_val,max_val)

        # Add y ticks
        for tick in ax.get_yticks():
            ax.axhline(tick,linestyle='-',color='k',zorder=20,alpha=0.3,linewidth=0.5)

        # Add plot legend
        ax.legend(bbox_to_anchor=(0., -.16, 1., -1.02), loc=3, ncol=4, mode="expand", borderaxespad=0.)

        # Format and add plot title
        plot_title = f'{self.attrs["name"]}, {self.attrs["state"]}\n'
        if year is not None:
            plot_title += f'{year} '
        if mode == 'min':
            plot_title += f'Minimum Temperature'
        elif mode == 'max':
            plot_title += f'Maximum Temperature'
        elif mode == 'average':
            plot_title += f'Average Temperature'
        else:
            plot_title += 'Temperature'
        if year is None:
            plot_title += ' Climatology'
        else:
            plot_title += 's'
        if year == keys[-1]:
            plot_title += f' (Through {dates[-1].strftime("%m/%d")})'
        a = ax.set_title(plot_title,loc='left',fontsize=12,fontweight='bold')
        right_title = f"Climatology: {climatology[0]} - {climatology[-1]}"
        right_title += f"\nExtremes: {years[0]} - {years[-1]}"
        ax.set_title(right_title,loc='right',fontsize=10)
        ax.text(0.99,0.01,'Image generated by pythreadex | Data from ThreadEx',
                ha='right', va='bottom', alpha=0.6, fontsize=9, transform=ax.transAxes)

        for spine in ax.spines.values():
            spine.set_zorder(10)

        return ax
    
    def plot_climatology_comparison(self, mode, climatology_start=None, climatology_end=None, date_range=None):
        r"""
        Plots a comparison of two climatologies.
        
        Parameters
        ----------
        mode : str
            Plotting mode. Can be "max", "min", or "average".
        climatology_start : tuple, optional
            Tuple with 2 elements denoting the start and end years of the starting climatology. If None (default), the earliest available 30-year climatology is used.
        climatology_end : tuple, optional
            Tuple with 2 elements denoting the start and end years of the ending climatology. If None (default), the latest available 30-year climatology is used.
        date_range : list or tuple, optional
            List or tuple with 2 elements denoting the start and end dates (e.g., ``['7/1', '11/30']``). If None (default), the full year is plotted.
        
        Returns
        -------
        matplotlib.pyplot.axes
            Axes instance containing the plot.
        
        Notes
        -----
        This plotting function ignores February 29th.
        """
        
        # Apply basic quality control to data
        if mode not in ['max', 'min', 'average']:
            raise ValueError('Acceptable values for "mode" are "max", "min" or "average".')
        if mode == 'average':
            data = self.filtered_data['both'].copy()
        else:
            data = self.filtered_data[mode].copy()
        if date_range is None:
            date_range = ['1/1', '12/31']

        # Determine start & end climatology ranges
        dates = [pd.to_datetime(i) for i in data['date']]
        years = np.unique([int(i.year) for i in dates])
        nearest_start_year = round_to_ten(years[0])+1
        if nearest_start_year-10 in years:
            nearest_start_year = nearest_start_year - 10
        if climatology_start is None:
            climatology_start = (nearest_start_year,nearest_start_year+29)
        nearest_end_year = round_to_ten(years[-2])-10
        if climatology_end is None:
            climatology_end = (nearest_end_year-29,nearest_end_year)
        
        # Calculate start & end climatologies
        climo_start = self.calc_temperature_climatology(mode, climatology_start)
        climo_end = self.calc_temperature_climatology(mode, climatology_end)
        
        # Create figure
        fig,ax = plt.subplots(figsize=(9,6),dpi=200)
        ax.grid(axis='y')

        # Plot month ticks and labels
        start_lim = dt.datetime.strptime(f'2015/{date_range[0]}','%Y/%m/%d')
        end_lim = dt.datetime.strptime(f'2015/{date_range[1]}','%Y/%m/%d')
        diff_lim = (end_lim-start_lim).total_seconds()/86400
        if diff_lim > 90:
            julian_data = months_in_julian(2015)
            xticks = julian_data['start']
            for i,(istart,iend) in enumerate(zip(xticks[::2],xticks[1:][::2])):
                ax.axvspan(istart,iend,color='#e4e4e4',alpha=0.5,zorder=0)
                ax.axvline(istart,linestyle='-',color='k',zorder=20,alpha=0.3,linewidth=0.5)
                ax.axvline(iend,linestyle='-',color='k',zorder=20,alpha=0.3,linewidth=0.5)
            ax.set_xticks(julian_data['midpoint'])
            ax.set_xticklabels(julian_data['name'])
        else:
            xticks = []
            xticklabels = []
            iter_lim = start_lim + dt.timedelta(hours=0)
            while iter_lim <= end_lim + dt.timedelta(hours=24*7):
                xticks.append(convert_to_julian(iter_lim))
                xticklabels.append(iter_lim.strftime('%b %d'))
                iter_lim += dt.timedelta(hours=24*7)
            for i,(istart,iend) in enumerate(zip(xticks[::2],xticks[1:][::2])):
                ax.axvspan(istart,iend,color='#e4e4e4',alpha=0.5,zorder=0)
                ax.axvline(istart,linestyle='-',color='k',zorder=20,alpha=0.3,linewidth=0.5)
                ax.axvline(iend,linestyle='-',color='k',zorder=20,alpha=0.3,linewidth=0.5)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
        ax.set_ylabel("Temperature (F)")
        
        # Plot climatological data
        color_store = {
            'max': {'start': '#FE7979', 'end': '#BD0A0A', 'mean_start': '#D20909', 'mean_end': '#750000'},
            'min': {'start': '#75B4FB', 'end': '#1668C2', 'mean_start': '#4343F9', 'mean_end': '#001295'},
            'average': {'start': '#B67EFD', 'end': '#7024D1', 'mean_start': '#9730DD', 'mean_end': '#6307A2'},
        }
        ax.fill_between(climo_start[mode]['days'],climo_start[mode]['p10'],climo_start[mode]['p90'],
                        color=color_store.get(mode)['start'],alpha=0.2,zorder=4,label='10-90th Percentile (Start)')
        ax.fill_between(climo_end[mode]['days'],climo_end[mode]['p10'],climo_end[mode]['p90'],
                        color=color_store.get(mode)['end'],alpha=0.2,zorder=4,label='10-90th Percentile (End)')
        ax.plot(climo_start[mode]['days'],climo_start[mode]['average'],color_store.get(mode)['mean_start'],
                zorder=5,linewidth=2.0,label=f'Average (Start)')
        ax.plot(climo_end[mode]['days'],climo_end[mode]['average'],color_store.get(mode)['mean_end'],
                zorder=5,linewidth=2.0,label=f'Average (End)')
        ax.plot(climo_start[mode]['days'],climo_start[mode]['p90'],color_store.get(mode)['start'],
                zorder=5,linewidth=0.5,linestyle='dashed')
        ax.plot(climo_start[mode]['days'],climo_start[mode]['p10'],color_store.get(mode)['start'],
                zorder=5,linewidth=0.5,linestyle='dashed')
        ax.plot(climo_end[mode]['days'],climo_end[mode]['p90'],color_store.get(mode)['end'],
                zorder=5,linewidth=0.5,linestyle='dashed')
        ax.plot(climo_end[mode]['days'],climo_end[mode]['p10'],color_store.get(mode)['end'],
                zorder=5,linewidth=0.5,linestyle='dashed')

        # Set limit of days
        start_count = int((start_lim - dt.datetime(2015,1,1)).total_seconds()/86400)
        end_count = int((end_lim - dt.datetime(2015,1,1)).total_seconds()/86400)
        ax.set_xlim(climo_start[mode]['days'][start_count],climo_start[mode]['days'][end_count])

        # Add plot legend
        ax.legend(bbox_to_anchor=(0., -.11, 1., -1.02), loc=3, ncol=4, mode="expand", borderaxespad=0.)

        # Format and add plot title
        plot_title = f'{self.attrs["name"]}, {self.attrs["state"]}\n'
        if mode == 'min':
            plot_title += f'Minimum Temperatures'
        elif mode == 'max':
            plot_title += f'Maximum Temperatures'
        else:
            plot_title += 'Average Temperatures'
        a = ax.set_title(plot_title,loc='left',fontsize=12)
        right_title = f"Start Climatology: {climatology_start[0]} - {climatology_start[1]}"
        right_title += f"\nEnd Climatology: {climatology_end[0]} - {climatology_end[1]}"
        ax.set_title(right_title,loc='right',fontsize=10)
        ax.text(0.99,0.01,'Image generated by pythreadex | Data from ThreadEx',
                ha='right', va='bottom', alpha=0.6, fontsize=9, transform=ax.transAxes)

        for spine in ax.spines.values():
            spine.set_zorder(10)
        
        return ax

    def plot_snow_time_series(self, season=None, rolling_days=None, climatology=None, climatology_years=30):
        r"""
        Plots a time series of station cumulative snowfall.
        
        Parameters
        ----------
        season : str, optional
            Start and end years of the requested season to plot (e.g., "1974-1975"). If None (default), no season is plotted.
        rolling_days : int, optional
            Number of days to compute a rolling climatology for. If None (default), cumulative snowfall is computed.
        climatology : tuple, optional
            Tuple with 2 elements denoting the start and end years to calculate climatology for. If None (default), this is automatically determined.
        climatology_years : int, optional
            Number of years to use for the default automatic climatology selection. Default is 30 years.
        
        Returns
        -------
        matplotlib.pyplot.axes
            Axes instance containing the plot.
        """

        # Retrieve data & perform basic QC
        data = self.data.copy()
        data['snow'] = data['snow'].replace('T',0.0)
        data['snow'] = data['snow'].astype(float)
        data = data[~np.isnan(data['snow'])]
        data = data[data['snow'] >= 0]
        if self.metric:
            data = data[data['snow'] <= 60 * 25.4]
        else:
            data = data[data['snow'] <= 60]
        snow = [i for i in data['snow'].values]
        dates = [dt.datetime.strptime(i,'%Y-%m-%d') for i in data['date'].values]

        # Get years in dataset
        years = np.unique([int(i.year) for i in dates])

        # Construct dict
        start_year = years[0]
        data = {}
        for i,(i_snow,i_date) in enumerate(zip(snow,dates)):
            if i_date.month not in [9,10,11,12,1,2,3,4,5,6]: continue
            if i_date < dt.datetime(start_year,10,1): continue

            # Format season string
            if i_date.month in [9,10,11,12]:
                season_format = f"{i_date.year}-{i_date.year+1}"
                season_start_date = dt.datetime((i_date.year),9,1)
                season_end_date = dt.datetime((i_date.year)+1,6,30)
                if calendar.isleap((i_date.year)+1): season_end_date = dt.datetime((i_date.year)+1,6,29)
            else:
                season_format = f"{i_date.year-1}-{i_date.year}"
                season_start_date = dt.datetime((i_date.year)-1,9,1)
                season_end_date = dt.datetime((i_date.year),6,30)
                if calendar.isleap(i_date.year): season_end_date = dt.datetime((i_date.year),6,29)
            if season_format not in data.keys():
                data[season_format] = {'days':[],'dates':[],'snow':[]}
                temp_start_date = season_start_date + dt.timedelta(hours=0)
                while temp_start_date <= season_end_date:
                    data[season_format]['days'].append((temp_start_date-season_start_date).total_seconds() / 86400)
                    data[season_format]['dates'].append(temp_start_date)
                    data[season_format]['snow'].append(0)
                    temp_start_date += dt.timedelta(hours=24)

            day_in_season = (i_date-season_start_date).total_seconds() / 86400
            if day_in_season == 0: continue
            if i_date > season_end_date: continue

            days_since_last_date = (i_date-dates[i-1]).total_seconds() / 86400
            idx = data[season_format]['dates'].index(i_date)
            if days_since_last_date == 1:
                data[season_format]['snow'][idx] = data[season_format]['snow'][idx-1] + i_snow
            else:
                for j in np.arange(days_since_last_date)[::-1]:
                    temp_date = i_date - dt.timedelta(days=j)
                    if temp_date in data[season_format]['dates']:
                        iter_idx = data[season_format]['dates'].index(temp_date)
                        data[season_format]['snow'][iter_idx] = data[season_format]['snow'][iter_idx-1]

        # Make sure cumulative snow is increasing
        for i_season in data.keys():
            for i in range(1,len(data[i_season]['snow'])):
                if data[i_season]['snow'][i] < data[i_season]['snow'][i-1]:
                    data[i_season]['snow'][i] = data[i_season]['snow'][i-1]

        #Calculate rolling data
        if rolling_days is not None:
            for i_season in data.keys():
                snow_start = np.array(data[i_season]['snow'])[rolling_days:]
                snow_end = np.array(data[i_season]['snow'])[:-rolling_days]
                data[i_season]['snow'] = [data[i_season]['snow'][i] for i in range(30)] + list(snow_start - snow_end)
        
        # Get year range for climatology
        if climatology is None:
            if season is None:
                nearest_end_year = round_to_ten(years[-1])-10
            else:
                nearest_end_year = round_to_ten(int(season.split('-')[0]))-10
            climatology = (nearest_end_year-climatology_years+1,nearest_end_year)
            if nearest_end_year-climatology_years+1 < years[0]:
                nearest_start_year = round_to_ten(years[0])+1
                if nearest_start_year-10 in years: nearest_start_year = nearest_start_year - 10
                climatology = (nearest_start_year,nearest_start_year+climatology_years+1)
        if climatology[1]-climatology[0] < climatology_years-1:
            raise ValueError(f"Error: Climatology must have at least {climatology_years} years.")
        
        # Calculate climatology
        keys = [k for k in data.keys()]
        climo = {
            'days': data[keys[-2]]['days'],
            'avg': [],
            'min': [],
            'p10': [],
            'p25': [],
            'p75': [],
            'p90': [],
            'max': []
        }
        for i in range(len(data[keys[-2]]['days'])):
            temp_array_full = []
            temp_array_climo = []
            for i_season in keys[:-1]:
                temp_array_full.append(data[i_season]['snow'][i])
                if climatology[0] <= int(i_season.split('-')[0]) <= climatology[-1]:
                    temp_array_climo.append(data[i_season]['snow'][i])
            climo['min'].append(np.nanmin(temp_array_full))
            climo['max'].append(np.nanmax(temp_array_full))
            climo['avg'].append(np.nanmean(temp_array_climo))
            climo['p10'].append(np.nanpercentile(temp_array_climo,10))
            climo['p25'].append(np.nanpercentile(temp_array_climo,25))
            climo['p75'].append(np.nanpercentile(temp_array_climo,75))
            climo['p90'].append(np.nanpercentile(temp_array_climo,90))
    
        # Create figure
        fig,ax = plt.subplots(figsize=(9,6),dpi=200)
        ax.grid(axis='y')

        # Decorate plot and label xticks
        xticks_data = [[2020,9],[2020,10],[2020,11],[2020,12],[2021,1],
                       [2021,2],[2021,3],[2021,4],[2021,5],[2021,6],[2021,7]]
        xticks = [(dt.datetime(i[0],i[1],1)-dt.datetime(2020,9,1)).total_seconds() / 86400 for i in xticks_data]
        for i,(istart,iend) in enumerate(zip(xticks[:-1][::2],xticks[1:][::2])):
            ax.axvspan(istart,iend,color='#e4e4e4',alpha=0.5,zorder=0)
            ax.axvline(istart,0,600,linestyle='-',color='k',alpha=0.3,linewidth=0.5)
            ax.axvline(iend,0,600,linestyle='-',color='k',alpha=0.3,linewidth=0.5)
        xticks = [(xticks[i]+xticks[i+1])/2.0 for i in range(len(xticks)-1)]
        xlabels = [(dt.datetime(2016,9,13) + dt.timedelta(hours=i*24)).strftime('%b') for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        unit_label = 'cm' if self.metric else 'inches'
        ax.set_ylabel(f'Snow ({unit_label})')
        ax.set_xlim(0,climo['days'][-1])

        # Plot climatology and extrema
        unit_label = ' cm' if self.metric else '"'
        ax.fill_between(climo['days'],climo['p10'],climo['p90'],color='#C8E4F9',label='10 - 90th Percentile')
        ax.fill_between(climo['days'],climo['p25'],climo['p75'],color='#8CC8F4',label='25 - 75th Percentile')
        ax.plot(climo['days'],climo['min'],'r',linewidth=1.0,linestyle='dashed',label=f'Min ({climo["min"][-1]:.1f}{unit_label})')
        ax.plot(climo['days'],climo['max'],'b',linewidth=1.0,linestyle='dashed',label=f'Max ({climo["max"][-1]:.1f}{unit_label})')
        ax.plot(climo['days'],climo['avg'],'b',linewidth=2.0,label=f'Average ({climo["avg"][-1]:.1f}{unit_label})')

        # Plot requested season
        if season is not None:
            keys = [k for k in data.keys()]
            if season == keys[-1]:
                idx = data[season]['dates'].index(dates[-1])
                ax.plot(data[season]['days'][:idx+1],data[season]['snow'][:idx+1],linewidth=2.5,color='k',
                        label=season+f' ({data[season]["snow"][-1]:.1f}{unit_label})')
                ax.plot(data[season]['days'][idx],data[season]['snow'][idx],'o',mfc='k',ms=8,mec='k',mew=0.5)
            else:
                ax.plot(data[season]['days'],data[season]['snow'],linewidth=2.5,color='k',
                            label=season+f' ({data[season]["snow"][-1]:.1f}{unit_label})')

        # Finalize plot
        ax.legend(loc='upper left')
        add_text = 'Cumulative Snowfall Climatology'
        if rolling_days is not None:
            add_text = f'{rolling_days}-Day Rolling Snowfall Climatology'
        a = ax.set_title(f'{self.attrs["name"]}, {self.attrs["state"]}\n{add_text}',
                         loc='left',fontsize=12)
        right_title = f"Climatology: {climatology[0]} - {climatology[1]}\n"
        right_title += f"Extremes: {years[0]} - {years[-1]}"
        ax.set_title(right_title, loc='right',fontsize=10)
        ax.text(0.99,0.01,'Image generated by pythreadex | Data from ThreadEx',
                alpha=0.6, fontsize=9, ha='right', va='bottom', transform=ax.transAxes)

        for spine in ax.spines.values():
            spine.set_zorder(10)
        
        return ax

    def plot_snowdepth_time_series(self, season=None, climatology=None, climatology_years=30):
        r"""
        Plots a time series of station snow depth.
        
        Parameters
        ----------
        season : str, optional
            Start and end years of the requested season to plot (e.g., "1974-1975"). If None (default), no season is plotted.
        climatology : tuple, optional
            Tuple with 2 elements denoting the start and end years to calculate climatology for. If None (default), this is automatically determined.
        climatology_years : int, optional
            Number of years to use for the default automatic climatology selection. Default is 30 years.
        
        Returns
        -------
        matplotlib.pyplot.axes
            Axes instance containing the plot.
        """

        # Retrieve data & perform basic QC
        data = self.data.copy()
        data['snow_depth'] = data['snow_depth'].replace('T',0.0)
        data['snow_depth'] = data['snow_depth'].astype(float)
        data = data[~np.isnan(data['snow_depth'])]
        data = data[data['snow_depth'] >= 0]
        snow = [i for i in data['snow_depth'].values]
        dates = [dt.datetime.strptime(i,'%Y-%m-%d') for i in data['date'].values]

        # Get years in dataset
        years = np.unique([int(i.year) for i in dates])

        # Construct dict
        start_year = years[0]
        data = {}
        for i,(i_snow,i_date) in enumerate(zip(snow,dates)):
            if i_date.month not in [9,10,11,12,1,2,3,4,5,6]: continue
            if i_date < dt.datetime(start_year,10,1): continue

            # Format season string
            if i_date.month in [9,10,11,12]:
                season_format = f"{i_date.year}-{i_date.year+1}"
                season_start_date = dt.datetime((i_date.year),9,1)
                season_end_date = dt.datetime((i_date.year)+1,6,30)
                if calendar.isleap((i_date.year)+1): season_end_date = dt.datetime((i_date.year)+1,6,29)
            else:
                season_format = f"{i_date.year-1}-{i_date.year}"
                season_start_date = dt.datetime((i_date.year)-1,9,1)
                season_end_date = dt.datetime((i_date.year),6,30)
                if calendar.isleap(i_date.year): season_end_date = dt.datetime((i_date.year),6,29)
            if season_format not in data.keys():
                data[season_format] = {'days':[],'dates':[],'snow':[]}
                temp_start_date = season_start_date + dt.timedelta(hours=0)
                while temp_start_date <= season_end_date:
                    data[season_format]['days'].append((temp_start_date-season_start_date).total_seconds() / 86400)
                    data[season_format]['dates'].append(temp_start_date)
                    data[season_format]['snow'].append(0)
                    temp_start_date += dt.timedelta(hours=24)

            day_in_season = (i_date-season_start_date).total_seconds() / 86400
            if day_in_season == 0: continue
            if i_date > season_end_date: continue

            days_since_last_date = (i_date-dates[i-1]).total_seconds() / 86400
            idx = data[season_format]['dates'].index(i_date)
            if days_since_last_date == 1:
                data[season_format]['snow'][idx] = i_snow
            else:
                for j in np.arange(days_since_last_date)[::-1]:
                    temp_date = i_date - dt.timedelta(days=j)
                    if temp_date in data[season_format]['dates']:
                        iter_idx = data[season_format]['dates'].index(temp_date)
                        data[season_format]['snow'][iter_idx] = np.nan

        # Get year range for climatology
        if climatology is None:
            if season is None:
                nearest_end_year = round_to_ten(years[-1])-10
            else:
                nearest_end_year = round_to_ten(int(season.split('-')[0]))-10
            climatology = (nearest_end_year-climatology_years+1,nearest_end_year)
            if nearest_end_year-climatology_years+1 < years[0]:
                nearest_start_year = round_to_ten(years[0])+1
                if nearest_start_year-10 in years: nearest_start_year = nearest_start_year - 10
                climatology = (nearest_start_year,nearest_start_year+climatology_years+1)
        if climatology[1]-climatology[0] < climatology_years-1:
            raise ValueError(f"Error: Climatology must have at least {climatology_years} years.")
        
        # Calculate climatology
        keys = [k for k in data.keys()]
        climo = {
            'days': data[keys[-2]]['days'],
            'avg': [],
            'min': [],
            'p10': [],
            'p25': [],
            'p75': [],
            'p90': [],
            'max': []
        }
        for i in range(len(data[keys[-2]]['days'])):
            temp_array_full = []
            temp_array_climo = []
            for i_season in keys[:-1]:
                temp_array_full.append(data[i_season]['snow'][i])
                if climatology[0] <= int(i_season.split('-')[0]) <= climatology[-1]:
                    temp_array_climo.append(data[i_season]['snow'][i])
            climo['min'].append(np.nanmin(temp_array_full))
            climo['max'].append(np.nanmax(temp_array_full))
            climo['avg'].append(np.nanmean(temp_array_climo))
            climo['p10'].append(np.nanpercentile(temp_array_climo,10))
            climo['p25'].append(np.nanpercentile(temp_array_climo,25))
            climo['p75'].append(np.nanpercentile(temp_array_climo,75))
            climo['p90'].append(np.nanpercentile(temp_array_climo,90))
    
        # Create figure
        fig,ax = plt.subplots(figsize=(9,6),dpi=200)
        ax.grid(axis='y')

        # Decorate plot and label xticks
        xticks_data = [[2020,9],[2020,10],[2020,11],[2020,12],[2021,1],
                       [2021,2],[2021,3],[2021,4],[2021,5],[2021,6],[2021,7]]
        xticks = [(dt.datetime(i[0],i[1],1)-dt.datetime(2020,9,1)).total_seconds() / 86400 for i in xticks_data]
        for i,(istart,iend) in enumerate(zip(xticks[:-1][::2],xticks[1:][::2])):
            ax.axvspan(istart,iend,color='#e4e4e4',alpha=0.5,zorder=0)
            ax.axvline(istart,0,600,linestyle='-',color='k',alpha=0.3,linewidth=0.5)
            ax.axvline(iend,0,600,linestyle='-',color='k',alpha=0.3,linewidth=0.5)
        xticks = [(xticks[i]+xticks[i+1])/2.0 for i in range(len(xticks)-1)]
        xlabels = [(dt.datetime(2016,9,13) + dt.timedelta(hours=i*24)).strftime('%b') for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        unit_label = 'cm' if self.metric else 'inches'
        ax.set_ylabel(f'Snow ({unit_label})')
        ax.set_xlim(0,climo['days'][-1])

        # Plot climatology and extrema
        unit_label = ' cm' if self.metric else '"'
        ax.fill_between(climo['days'],climo['p10'],climo['p90'],color='#C8E4F9',label='10 - 90th Percentile')
        ax.fill_between(climo['days'],climo['p25'],climo['p75'],color='#8CC8F4',label='25 - 75th Percentile')
        ax.plot(climo['days'],climo['min'],'r',linewidth=1.0,linestyle='dashed',label=f'Min ({max(climo["min"]):.1f}{unit_label})')
        ax.plot(climo['days'],climo['max'],'b',linewidth=1.0,linestyle='dashed',label=f'Max ({max(climo["max"]):.1f}{unit_label})')
        ax.plot(climo['days'],climo['avg'],'b',linewidth=2.0,label=f'Average ({max(climo["avg"]):.1f}{unit_label})')

        # Plot requested season
        if season is not None:
            keys = [k for k in data.keys()]
            if season == keys[-1]:
                idx = data[season]['dates'].index(dates[-1])
                ax.plot(data[season]['days'][:idx+1],data[season]['snow'][:idx+1],linewidth=2.5,color='k',
                        label=season+f' ({max(data[season]["snow"]):.1f}{unit_label})')
                ax.plot(data[season]['days'][idx],data[season]['snow'][idx],'o',mfc='k',ms=8,mec='k',mew=0.5)
            else:
                ax.plot(data[season]['days'],data[season]['snow'],linewidth=2.5,color='k',
                            label=season+f' ({max(data[season]["snow"]):.1f}{unit_label})')

        # Finalize plot
        ax.legend(loc='upper left')
        a = ax.set_title(f'{self.attrs["name"]}, {self.attrs["state"]}\nSnow Depth Climatology',
                         loc='left',fontsize=12)
        right_title = f"Climatology: {climatology[0]} - {climatology[1]}\n"
        right_title += f"Extremes: {years[0]} - {years[-1]}"
        ax.set_title(right_title, loc='right',fontsize=10)
        ax.text(0.99,0.01,'Image generated by pythreadex | Data from ThreadEx',
                alpha=0.6, fontsize=9, ha='right', va='bottom', transform=ax.transAxes)

        for spine in ax.spines.values():
            spine.set_zorder(10)
        
        return ax

def round_to_ten(x):
    
    return int(np.ceil(x / 10.0)) * 10

def add_rectangle(ax,x,vmin,vmax,width=0.6,color='b',alpha=0.1,linewidth=0.6):
    
    llc = x-(width/2.0)
    rectangle = plt.Rectangle((llc,vmin),width,(vmax-vmin),fc=color,alpha=alpha,ec='k',linewidth=linewidth,zorder=7)
    ax.add_patch(rectangle)
    return ax

def convert_to_julian(time):

    year = time.year
    return ((time - dt.datetime(year,1,1,0)).days + (time - dt.datetime(year,1,1,0)).seconds/86400.0)

def months_in_julian(year):

    #Get number of days in year
    length_of_year = convert_to_julian(dt.datetime(year,12,31,0))+1.0
    
    #Construct a list of months and names
    months = range(1,13,1)
    months_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    months_dates = [dt.datetime(year,i,1,0) for i in months]
    
    #Get midpoint x-axis location of month
    months_julian = [int(convert_to_julian(i)) for i in months_dates]
    midpoint_julian = (np.array(months_julian) + np.array(months_julian[1:]+[length_of_year]))/2.0
    return {'start':months_julian,'midpoint':midpoint_julian.tolist(),'name':months_names}
