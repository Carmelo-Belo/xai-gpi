from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import itertools
import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
import os



def filter_xarray(data, min_lat=-90, max_lat=90, min_lon=-180, max_lon=180, months=[1,2,3,4,5,6,7,8,9,10,11,12],resolution=2):
    
    ''' This function filters the data by latitude and longitude'''
    
    #Filtering the data
    filtered_dataset = data.sel(latitude=np.arange(max_lat, min_lat-resolution,-resolution), longitude=np.arange(min_lon, max_lon+resolution,resolution))

    filtered_dataset = filtered_dataset.sel(time=filtered_dataset['time.month'].isin(months))
    
    return filtered_dataset

def seasonal_smoothing(data_filtered_clima, variable, data_filtered):
    year_average = data_filtered_clima.groupby('time.dayofyear').mean('time')
    year_average2 = np.append(np.append(year_average[variable].values, year_average[variable].values,axis=0), year_average[variable].values,axis=0)
    year_average_xarray = xr.DataArray(data=year_average2,dims=["dayofyear", "latitude", "longitude"],)
    year_average_smooth = year_average.rolling(dayofyear=30,min_periods=1, center=True).mean('time')
    year_average_smooth[variable] = year_average_xarray.rolling(dayofyear=30,min_periods=1, center=True).mean('time')[366:732,:,:]
    year_average_smooth_nonleap = year_average_smooth.sel(dayofyear=year_average_smooth['dayofyear']!=60)

    years = data_filtered.groupby('time.year').mean().year.values

    import calendar

    for year in years:
        is_leap_year = calendar.isleap(year)
        year_data = data_filtered.sel(time=data_filtered['time.year'] == year)

        if is_leap_year:
            diff = year_data[variable].values - year_average_smooth[variable].values
        else:
            diff = year_data[variable].values - year_average_smooth_nonleap[variable].values
        year_data[variable] = (('time', 'latitude', 'longitude'), diff)  
        data_filtered[variable].loc[dict(time=data_filtered['time.year'] == year)] = year_data[variable].values

    return data_filtered

def perform_clustering(var, level, months, basin, n_clusters, norm, seasonal_soothing, train_yearI, train_yearF, test_yearI, test_yearF,
                       resolution, path_predictor, path_output):

    # Define geographical coordinates according to the basin considered
    if basin == 'NWP':
        min_lon, max_lon, min_lat, max_lat = 100, 180, 0, 40
    elif basin == 'NEP':
        min_lon, max_lon, min_lat, max_lat = -180, -75, 0, 40
    elif basin == 'NA':
        min_lon, max_lon, min_lat, max_lat = -100, 0, 0, 40
    elif basin == 'NI':
        min_lon, max_lon, min_lat, max_lat = 45, 100, 0, 40
    elif basin == 'SP':
        min_lon, max_lon, min_lat, max_lat = 135, -70, -40, 0
    elif basin == 'SI':
        min_lon, max_lon, min_lat, max_lat = 35, 135, -40, 0
    elif basin == 'GLB':
        min_lon, max_lon, min_lat, max_lat = -181, 181, -40, 40
    else:
        raise ValueError('Basin not recognized')

    # Data extraction from .nc files
    for y, year in enumerate(range(train_yearI, train_yearF+1)):
        path = path_predictor + f'_{resolution}_{year}.nc'
        if y == 0:
            train_data = xr.open_dataset(path)[var]
        else:
            train_data = xr.concat([train_data, xr.open_dataset(path)[var]], dim='time')
    for y, year in enumerate(range(test_yearI, test_yearF+1)):
        path = path_predictor + f'_{resolution}_{year}.nc'
        if y == 0:
            test_data = xr.open_dataset(path)[var]
        else:
            test_data = xr.concat([test_data, xr.open_dataset(path)[var]], dim='time')
    # If variable is defined on pressure levels, select the level specified in the inputs
    if (level != None) and (type(level) == int):
        train_data = train_data.sel(level=level)
        test_data = test_data.sel(level=level)
        var_name = train_data.long_name + ' at ' + str(level) + ' hPa'
        var = var + str(level)
    # If variable is defined between the difference of two pressure levels, select the difference level specified in the inputs
    elif (level != None) and (type(level) == str):
        train_data = train_data.sel(diff_level=level)
        test_data = test_data.sel(diff_level=level)
        var_name = train_data.long_name + ' between ' + level.split('-')[1] + ' and ' + level.split('-')[0] + ' hPa'
        var = var + level
    else:
        var_name = train_data.long_name
    total_data = xr.concat([train_data, test_data], dim='time')

    ## Perform the cluster only on the train years
    # Data preprocessing
    # from clustering import filter_xarray

    ## CHECK BETTER FILTERING PROCESS WHEN WORKING BASIN WISE ##
    # Data is filtered based on the geographical limits, months, resolution and years
    # data_filtered = filter_xarray(daily_data_train, min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon, months=months,resolution=resolution)
    # data_filtered_clima = filter_xarray(data_clima_time, min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon, months=months,resolution=resolution)

    # Perform the seasonal soothing
    ## SEASONAL SMOOTHING IS DONE ON DAYS OF YEAR, NOT ON MONTHS ##
    # from clustering import seasonal_smoothing
    # if seasonal_soothing == True:
    #     data_filtered = seasonal_smoothing(data_filtered_clima, variable, data_filtered)

    # Reshape the data -> (time, lat, lon) -> (lat*lon, time)
    data = train_data.values
    data_res = data.reshape(data.shape[0], data.shape[1]*data.shape[2]).T

    # Mask the data if is a variable defined over the ocean
    ocean_vars = ['sst', 'ssta20', 'ssta30', 'mpi']
    if var in ocean_vars:
        ocean_mask = ~np.any(np.isnan(data_res), axis=1)
    else:
        ocean_mask = None
    ## MASKING ONLY FOR OCEAN VARIABLES, NEED TO CHECK IF NECESSARY FOR OTHER VARIABLES, OR WHEN WORKING BASIN WISE ##
    mask = ocean_mask
    if mask is None:
        data_res_masked = data_res
    else:
        data_res_masked = data_res[mask]

    # Normalize each time series
    if norm==True:
        data_res_masked = normalize(data_res_masked, axis=1, copy=True, return_norm=False)

    # Perform the clustering
    from clustering import cluster_model
    cluster = cluster_model(data_res_masked, n_clusters, var)
    cluster.check_data()
    cluster.kmeans()
    # cluster.agclustering()

    # Get the closest node to the cluster center
    centroids = cluster_model.get_closest2center2(cluster, data_res_masked)

    # Plot the clusters
    latitudes = train_data.latitude.values
    longitudes = train_data.longitude.values
    cluster_model.plot_clusters(cluster, data_res_masked, latitudes, longitudes, mask, var_name)

    # Get the data for the centroids 
    iter = itertools.product(latitudes, longitudes)
    nodes_list = list(iter)
    if mask is None:
        nodes_list = np.array(nodes_list)
    else:
        nodes_list = np.array(nodes_list)[mask]

    lons_c = [np.array(nodes_list)[centroids][i][1] for i in range(len(np.array(nodes_list)[centroids]))]
    lats_c = [np.array(nodes_list)[centroids][i][0] for i in range(len(np.array(nodes_list)[centroids]))]

    # Once the cluster are created, read and process the test data
    ## LOOK BETTER AT FILTERING WHEN PERFORMING THE BASIN WISE ANALYSIS ## 
    # data_filtered_test = filter_xarray(daily_data_test, min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon, months=months, resolution=resolution)
    
    # Apply seasonal forecasting
    ## SEASONAL SMOOTHING IS DONE ON DAYS OF YEAR, NOT ON MONTHS ##
    # if seasonal_soothing == True:
    #     data_filtered_test = seasonal_smoothing(data_filtered_clima, variable, data_filtered_test)
        
    # Merge the train and test data
    # data_filtered_total = xr.concat([data_filtered, data_filtered_test], dim='time')
    ## FOR NOW THERE IS NO FILTERING OF THE DATA, SO data_filtered_total = total_data ##
    data_filtered_total = total_data

    # Create a dataframe with the centroids timeseries
    centroids_data = []
    for i in range(len(centroids)):
        centroid_data = data_filtered_total.sel(latitude=lats_c[i], longitude=lons_c[i]).values
        centroids_data.append(centroid_data)
    centroids_dataframe = pd.DataFrame(centroids_data).T
    centroids_dataframe.index = data_filtered_total.time.values
    centroids_dataframe.columns = [var + basin + '_cluster' + str(i) for i in range(1, n_clusters+1)]

    # Get average data for each cluster, weighted averages are calculated. Batch size is adjusted to avoid memory errors
    clusters_av_dataframe = pd.DataFrame(columns=[var + basin + '_cluster' + str(i) for i in range(1, n_clusters+1)])
    weights = np.cos(np.deg2rad(nodes_list[:,0]))
    data_cluster_avg = data_filtered_total.values

    def weighted_average(data, weights):
        weighted_sum = np.dot(weights, data)
        total_weight = np.sum(weights)
        return weighted_sum / total_weight

    def calculate_weighted_average(data, weights, batch_size):
        num_rows = data.shape[0]
        result = np.zeros((data.shape[1],))
        
        for i in range(0, num_rows, batch_size):
            if i + batch_size > num_rows:
                batch_data = data[i:]
                batch_weights = weights[i:]
            else:
                batch_data = data[i:i+batch_size]
                batch_weights = weights[i:i+batch_size]
            result += weighted_average(batch_data, batch_weights)
            
        return result / (num_rows/batch_size)

    for c in range(len(centroids)):
        cluster_mask = cluster.labels == c
        batch_size = 100
        if mask is None:
            data_cluster_avg_masked = data_cluster_avg.reshape(data_cluster_avg.shape[0], data_cluster_avg.shape[1]*data_cluster_avg.shape[2]).T[cluster_mask]
        else:
            data_cluster_avg_masked = data_cluster_avg.reshape(data_cluster_avg.shape[0], data_cluster_avg.shape[1]*data_cluster_avg.shape[2]).T[mask][cluster_mask]
        weights_masked = weights[cluster_mask]
        cluster_avg = calculate_weighted_average(data_cluster_avg_masked, weights_masked, batch_size)
        clusters_av_dataframe[var + basin + '_cluster' + str(c+1)] = cluster_avg

    clusters_av_dataframe.index = data_filtered_total.time.values

    # Create a dataframe with the cluster labels
    labels_dataframe = pd.DataFrame(cluster.labels, columns=['cluster'])
    labels_dataframe['nodes_lat'] = np.array(nodes_list)[:,0]
    labels_dataframe['nodes_lon'] = np.array(nodes_list)[:,1]
    labels_dataframe['cluster'] = labels_dataframe['cluster'] + 1

    # Save the data
    centroids_dataframe.to_csv(os.path.join(path_output, 'centroids_' + var + basin + str(n_clusters) + '.csv'))
    clusters_av_dataframe.to_csv(os.path.join(path_output, 'averages_' + var + basin + str(n_clusters) + '.csv'))
    labels_dataframe.to_csv(os.path.join(path_output, 'labels_' + var + basin + str(n_clusters) + '.csv'))

    return centroids, centroids_dataframe, clusters_av_dataframe, labels_dataframe


def compute_ENSO(path_predictors,path_output,first_year,last_year, first_clima,last_clima,resolution):
    var = 'sst'

    import xarray as xr
    daily_data_train = xr.open_dataset(path_predictors+'data_daily_'+var+'_1950_2010.nc')
    daily_data_test = xr.open_dataset(path_predictors+'data_daily_'+var+'_2011_2022.nc')
    daily_data_total = xr.concat([daily_data_train, daily_data_test], dim='time')

    min_lat=-5
    max_lat=5
    min_lon=-170
    max_lon=-120

    # Perform the cluster only on the train years
    daily_data_train = daily_data_total.sel(time=slice(str(first_year)+'-01-01', str(int(last_year))+'-12-31'))
    data_clima_time = daily_data_total.sel(time=slice(str(first_clima)+'-01-01', str(int(last_clima))+'-12-31'))
    daily_data_total.close()

    variable = var  
    # Data preprocessing
    from FS_TCG.clustering.clustering import filter_xarray
    import numpy as np
    # Data is filtered based on the geographical limits, months, resolution and years
    data_filtered = filter_xarray(daily_data_train, min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon,resolution=resolution)
    data_filtered_clima = filter_xarray(data_clima_time, min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon,resolution=resolution)
    # data_filtered_test = filter_xarray(daily_data_test, min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon,resolution=resolution)

    # Perform the seasonal soothing
    
    from FS_TCG.clustering.clustering import seasonal_smoothing
    data_filtered = seasonal_smoothing(data_filtered_clima,variable,data_filtered)
    # data_filtered_test = seasonal_smoothing(data_filtered_clima,variable,data_filtered_test)

            
    data_filtered_clima.close()


    # Merge the train and test data

    # data_filtered_total = xr.concat([data_filtered, data_filtered_test], dim='time')
    data_filtered_total = data_filtered

    # Compute ENSO index

    enso = data_filtered_total.mean(dim=['latitude','longitude'])

    # Save the ENSO index

    enso = enso.to_dataframe()
    enso.columns = ['ENSO']
    enso.to_csv(path_output+'ENSO_index.csv')

    # Plot the ENSO index

    plt.plot(np.arange(0,len(enso)), enso['ENSO'])

    return enso


def compute_IOD(path_predictors,path_output,first_year,last_year, first_clima,last_clima,resolution):
    var = 'sst'

    # Compute area averaged SST anomaly in the western tropical Indian Ocean

    min_lat=-10
    max_lat=10
    min_lon=50
    max_lon=70

    import xarray as xr
    daily_data_train = xr.open_dataset(path_predictors+'data_daily_'+var+'_1950_2010.nc')
    daily_data_test = xr.open_dataset(path_predictors+'data_daily_'+var+'_2011_2022.nc')
    daily_data_total = xr.concat([daily_data_train, daily_data_test], dim='time')


    # Perform the cluster only on the train years
    daily_data_train = daily_data_total.sel(time=slice(str(first_year)+'-01-01', str(int(last_year))+'-12-31'))
    data_clima_time = daily_data_total.sel(time=slice(str(first_clima)+'-01-01', str(int(last_clima))+'-12-31'))
    daily_data_total.close()

    variable = var  
    # Data preprocessing
    from FS_TCG.clustering.clustering import filter_xarray
    import numpy as np
    # Data is filtered based on the geographical limits, months, resolution and years
    data_filtered = filter_xarray(daily_data_train, min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon,resolution=resolution)
    data_filtered_clima = filter_xarray(data_clima_time, min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon,resolution=resolution)
    # data_filtered_test = filter_xarray(daily_data_test, min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon,resolution=resolution)

    # Perform the seasonal soothing
    
    from FS_TCG.clustering.clustering import seasonal_smoothing
    data_filtered = seasonal_smoothing(data_filtered_clima,variable,data_filtered)
    # data_filtered_test = seasonal_smoothing(data_filtered_clima,variable,data_filtered_test)

            
    data_filtered_clima.close()
    # Merge the train and test data

    # data_filtered_total_1 = xr.concat([data_filtered, data_filtered_test], dim='time')
    data_filtered_total_1 = data_filtered
    # Compute area averaged SST anomaly in the sotheastern tropical Indian Ocean

    min_lat=-10
    max_lat=0
    min_lon=90
    max_lon=110

    import xarray as xr
    daily_data_train = xr.open_dataset(path_predictors+'data_daily_'+var+'_1950_2010.nc')
    daily_data_test = xr.open_dataset(path_predictors+'data_daily_'+var+'_2011_2022.nc')
    


    # Perform the cluster only on the train years
    daily_data_train = daily_data_train.sel(time=slice(str(first_year)+'-01-01', str(int(last_year))+'-12-31'))
    data_clima_time = daily_data_train.sel(time=slice(str(first_clima)+'-01-01', str(int(last_clima))+'-12-31'))


    variable = var  
    # Data preprocessing
    from FS_TCG.clustering.clustering import filter_xarray
    import numpy as np
    # Data is filtered based on the geographical limits, months, resolution and years
    data_filtered = filter_xarray(daily_data_train, min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon,resolution=resolution)
    data_filtered_clima = filter_xarray(data_clima_time, min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon,resolution=resolution)
    data_filtered_test = filter_xarray(daily_data_test, min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon,resolution=resolution)

    # Perform the seasonal soothing
    
    from FS_TCG.clustering.clustering import seasonal_smoothing
    data_filtered = seasonal_smoothing(data_filtered_clima,variable,data_filtered)
    data_filtered_test = seasonal_smoothing(data_filtered_clima,variable,data_filtered_test)

            
    data_filtered_clima.close()
    # Merge the train and test data

    data_filtered_total_2 = xr.concat([data_filtered, data_filtered_test], dim='time')

    # Compute IOD index

    iod = data_filtered_total_1.mean(dim=['latitude','longitude'])-data_filtered_total_2.mean(dim=['latitude','longitude'])

    # Save the IOD index

    iod = iod.to_dataframe()
    iod.columns = ['IOD']
    iod.to_csv(path_output+'IOD_index.csv')

    # Plot the IOD index

    plt.plot(np.arange(0,len(iod)), iod['IOD'])

    return iod


class cluster_model:

    def __init__(self, data, n_clusters, name):
        self.data = data
        self.n_clusters = n_clusters
        self.name = name
        self.correct_data_shape = True
        self.labels = None
        self.cluster_centers = None
        self.silhouette_score = None

    def check_data(self):
        if isinstance(self.data, np.ndarray) and len(self.data.shape) == 2:
            print('Data is a 2D numpy array')
            print('Please, be sure the data is in the correct format: (n_samples (nodes), n_features (variables (time))')
            self.correct_data_shape = True
        else:
            print('Data is not a 2D numpy array')
            self.correct_data_shape = False
        
    def kmeans(self):
        self.cluster = KMeans(n_clusters=self.n_clusters, random_state=0, init='k-means++', n_init=10, tol=0.0001)
        self.cluster.fit(self.data)
        self.labels = self.cluster.labels_
        # self.silhouette_score = silhouette_score(data, self.labels)
        self.cluster_centers = self.cluster.cluster_centers_
        self.silhouette_score = silhouette_score(self.data, self.labels)
        # self.cluster_centers = self.cluster.cluster_centers_
        # self.predictions = self.cluster.predict(data)

    def agclustering(self):
        self.cluster = AgglomerativeClustering(n_clusters=self.n_clusters)
        self.cluster.fit(self.data)
        self.labels = self.cluster.labels_
        

    def dendogram(self, method='single', metric='euclidean' ):
        self.linkage_matrix = linkage(self.data, method=method, metric=metric)
        plt.figure(figsize=(20, 15))
        plt.title("Dendrograms")
        dendrogram(self.linkage_matrix)
        plt.show()

    def get_closest2center2(self, data):
        index = [np.argmin(np.linalg.norm(data[self.labels==i] - self.cluster_centers[i], ord=2, axis=1)) for i in range(self.cluster_centers.shape[0])]
        absolute_indexes = [np.where(self.labels == i)[0][closest_index] for i, closest_index in enumerate(index)]
        # print('Index of the closest cluster center for each sample', absolute_indexes)
        return absolute_indexes
    
    def get_closest2center(data, cluster_centers, labels):
        index = [np.argmin(np.linalg.norm(data[labels == i] - cluster_centers[i], ord=2, axis=1)) for i in range(cluster_centers.shape[0])]
        index_data = [np.argmin(np.linalg.norm(data[index[i]] - data, ord=2, axis=1)) for i in range(len(index))]
        print(index_data)
        return index_data

    def get_mean_clusters(self):
        mean_clusters = np.zeros((self.n_clusters))
        for i in range(self.n_clusters):
            mean_clusters[i] = np.mean(self.data[self.labels==i], axis=0)
        return mean_clusters

    def plot_clusters(cluster, data, latitudes, longitudes, mask, title, basin='GLB'):
        ## FOR NOW ONLY PLOT ON GLOBAL MAP, NEED TO ADAPT FOR BASIN WISE PLOTTING -> mask and basin##
        north, south = latitudes[0], latitudes[-1]
        west, east = longitudes[0], longitudes[-1]
        fig = plt.figure(figsize=(30, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))

        # Set the extent of the map
        ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
        ax.coastlines(resolution='110m', linewidth=2)
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # Set the gridlines of the map 
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 20))
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 10))
        gl.xlabel_style = {'size': 20} 
        gl.ylabel_style = {'size': 20}
        
        # Get data for the plot and plot the clusters
        iter = itertools.product(latitudes, longitudes)
        if mask is None:
            nodes_list = np.array(list(iter))
        else:
            nodes_list = np.array(list(iter))[mask]
        lons = np.array([nodes_list[i][1] for i in range(len(nodes_list))])
        lats = np.array([nodes_list[i][0] for i in range(len(nodes_list))])
        n_clusters = len(np.unique(cluster.labels))
        cmap = plt.cm.get_cmap('tab20', n_clusters)
        scatter = ax.scatter(lons, lats, c=cluster.labels, cmap=cmap, s=400, transform=ccrs.PlateCarree())
        
        # Add colorbar
        bounds = np.arange(n_clusters + 1) - 0.5
        cbar = plt.colorbar(scatter, ticks=np.arange(n_clusters), boundaries=bounds, ax=ax, orientation='vertical')
        cbar.set_ticklabels(np.arange(n_clusters)+1)
        cbar.ax.tick_params(labelsize=22)
        cbar.set_label('Cluster',fontsize=26)
        
        # Plot the centroids witht the Cross symbol and number of the cluster next to it
        centroids = cluster_model.get_closest2center2(cluster, data)
        for c, centroid in enumerate(centroids):
            lon_c = nodes_list[centroid][1]
            lat_c = nodes_list[centroid][0]
            ax.scatter(lon_c, lat_c, marker='x', linewidth=4, s=500, color='black', transform=ccrs.PlateCarree())
            ax.text(lon_c+2.5, lat_c, str(c+1), fontsize=20, color='black', transform=ccrs.PlateCarree(),
                    ha='left', va='top')
        
        ax.set_title(title, fontsize=30)
        plt.tight_layout()
        plt.show()
