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

    def plot_clusters(cluster, data, latitudes, longitudes, mask, title):
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
        cmap = plt.get_cmap('tab20', n_clusters)
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
        return fig

def crop_field(var, lon1, lon2, lat1, lat2):
    """
    Crop the specified variable to the specified domain.

    Inputs:
        var: xarray.DataArray or xarray.Dataset
            The variable to crop
        lon1: float
            The western longitude of the domain (range: -180 to 180)
        lon2: float
            The eastern longitude of the domain (range: -180 to 180)
        lat1: float
            The southern latitude of the domain
        lat2: float
            The northern latitude of the domain
    Outputs:
        ds: xarray.DataArray or xarray.Dataset
            The cropped variable
    """
    ds = var.copy()
    # If the domain crosses/touches the meridian 180, convert to 0-360
    if (lon1 <= 180 and lon2 >= -180 and lon1 > lon2) or (lon1 == -180) or (lon2 == 180):
        ds.coords['longitude'] = (ds.longitude + 360) %360
        ds = ds.sortby(ds.longitude)
        if lon2 < 0:
            lon2 = lon2 + 360
        if lon1 < 0:
            lon1 = lon1 + 360
    return ds.sel(longitude=slice(lon1, lon2), latitude=slice(lat2, lat1))

# Functions to compute the weighted average of the data in the clusters
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

def perform_clustering(var, level, months, basin, n_clusters, norm, train_yearI, train_yearF, resolution, path_predictor, path_output, by_anomaly):
    """
    Perform clustering of the specified atmospheric variable.

    Inputs:
        var: str
            The acronym of the variable to cluster as saved in the .nc files
        level: str
            The pressure level of the variable to cluster, if surface level, set to 'sfc'
        months: list
            The months to consider for the clustering
        basin: str
            The basin considered for the clustering
        n_clusters: int
            The number of clusters to create
        norm: bool
            If True, normalize the data
        train_yearI: int
            The initial year for training
        train_yearF: int
            The final year for training
        resolution: str
            The resolution of the data
        path_predictor: str
            The path to the .nc files containing the data
        path_output: str
            The path to the output directory where to save the clustering results
        by_anomaly: bool
            If True, cluster according to the anomalies of the variable
    Outputs:
        centroids: list
            The indices of the nodes closest to the cluster centers
        centroids_dataframe: pd.DataFrame
            The dataframe containing the timeseries of the centroids
        clusters_av_dataframe: pd.DataFrame
            The dataframe containing the average timeseries of each cluster
        labels_dataframe: pd.DataFrame
            The dataframe containing the cluster labels of each node
    """

    ## Load the variable data to cluster ## 
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
    for y, year in enumerate(range(1965, 2023)):
        path = path_predictor + f'_{resolution}_{year}.nc'
        if y == 0:
            total_data = xr.open_dataset(path)[var]
        else:
            total_data = xr.concat([total_data, xr.open_dataset(path)[var]], dim='time')
    # If variable is defined on pressure levels, select the level specified in the inputs
    if (level != 'sfc') and (len(level) < 5):
        level = int(level)
        total_data = total_data.sel(level=level)
        var_name = total_data.long_name + ' at ' + str(level) + ' hPa'
        var = var + str(level)
    # If variable is defined between the difference of two pressure levels, select the difference level specified in the inputs
    elif (level != 'sfc') and (len(level) > 4):
        total_data = total_data.sel(diff_level=level)
        var_name = total_data.long_name + ' between ' + level.split('-')[1] + ' and ' + level.split('-')[0] + ' hPa'
        var = var + level
    else:
        var_name = total_data.long_name
    # Convert the data of some variables to the desired units
    if var == 'sst': # Convert from K to C
        total_data = total_data - 273.15
    elif var == 'msl': # Convert from Pa to hPa
        total_data = total_data / 100
    # Get the anomalies of the variable (group by month because we are working with monthly data)
    climatology = total_data.groupby('time.month').mean('time') 
    anomaly = (total_data.groupby('time.month') - climatology).drop('month')
    # Filtered data based on the geographical limits
    if basin != 'GLB':
        total_data = crop_field(total_data, min_lon, max_lon, min_lat, max_lat)
        anomaly = crop_field(anomaly, min_lon, max_lon, min_lat, max_lat)
    ## IF WE WANT TO WORK ONLY ON CYCLONE SEASON WHEN BASIN WISE WE NEED TO ADJUST IT HERE ##

    ## Perform the cluster only on the train years ##
    # Get the train data and the anomalies train data for the variable
    train_data = total_data.sel(time=slice(str(train_yearI)+'-01-01', str(train_yearF)+'-12-31'))
    train_data_anomaly = anomaly.sel(time=slice(str(train_yearI)+'-01-01', str(train_yearF)+'-12-31'))
    # Reshape the data -> (time, lat, lon) -> (lat*lon, time)
    data = train_data.values
    data_anomaly = train_data_anomaly.values
    data_res = data.reshape(data.shape[0], data.shape[1]*data.shape[2]).T
    data_res_anomaly = data_anomaly.reshape(data_anomaly.shape[0], data_anomaly.shape[1]*data_anomaly.shape[2]).T

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
        anomaly_res_masked = data_res_anomaly
    else:
        data_res_masked = data_res[mask]
        anomaly_res_masked = data_res_anomaly[mask]

    # Normalize each time series
    if norm==True:
        data_res_masked = normalize(data_res_masked, axis=1, copy=True, return_norm=False)
        anomaly_res_masked = normalize(anomaly_res_masked, axis=1, copy=True, return_norm=False)

    # Perform the clustering
    if by_anomaly == True:
        cluster = cluster_model(anomaly_res_masked, n_clusters, var)
        cluster.check_data()
        cluster.kmeans()
        # Get the closest node to the cluster center
        centroids = cluster_model.get_closest2center2(cluster, anomaly_res_masked)
    else:
        cluster = cluster_model(data_res_masked, n_clusters, var)
        cluster.check_data()
        cluster.kmeans()
        # Get the closest node to the cluster center
        centroids = cluster_model.get_closest2center2(cluster, data_res_masked)

    # Plot the clusters
    latitudes = train_data.latitude.values
    longitudes = train_data.longitude.values
    if by_anomaly == True:
        clusters_fig = cluster_model.plot_clusters(cluster, anomaly_res_masked, latitudes, longitudes, mask, var_name + ' anomaly')
    else:
        clusters_fig = cluster_model.plot_clusters(cluster, data_res_masked, latitudes, longitudes, mask, var_name)

    # Save the clusters figures in the output directory
    output_figs_dir = os.path.join(path_output, f'figures')
    os.makedirs(output_figs_dir, exist_ok=True)
    fig_name = os.path.join(output_figs_dir, f'{var}.pdf')
    clusters_fig.savefig(fig_name, bbox_inches='tight', format='pdf', dpi=300)
    plt.close(clusters_fig)

    # Get the data for the centroids 
    iter = itertools.product(latitudes, longitudes)
    nodes_list = list(iter)
    if mask is None:
        nodes_list = np.array(nodes_list)
    else:
        nodes_list = np.array(nodes_list)[mask]

    lons_c = [np.array(nodes_list)[centroids][i][1] for i in range(len(np.array(nodes_list)[centroids]))]
    lats_c = [np.array(nodes_list)[centroids][i][0] for i in range(len(np.array(nodes_list)[centroids]))]

    # Create a dataframe with the centroids timeseries
    centroids_data = []
    for i in range(len(centroids)):
        centroid_data = total_data.sel(latitude=lats_c[i], longitude=lons_c[i]).values
        centroids_data.append(centroid_data)
    centroids_dataframe = pd.DataFrame(centroids_data).T
    centroids_dataframe.index = total_data.time.values
    centroids_dataframe.columns = [var + '_cluster' + str(i) for i in range(1, n_clusters+1)]

    # Get average data for each cluster, weighted averages are calculated. Batch size is adjusted to avoid memory errors
    clusters_av_dataframe = pd.DataFrame(columns=[var + '_cluster' + str(i) for i in range(1, n_clusters+1)])
    weights = np.cos(np.deg2rad(nodes_list[:,0]))
    data_cluster_avg = total_data.values

    for c in range(len(centroids)):
        cluster_mask = cluster.labels == c
        batch_size = 100
        if mask is None:
            data_cluster_avg_masked = data_cluster_avg.reshape(data_cluster_avg.shape[0], data_cluster_avg.shape[1]*data_cluster_avg.shape[2]).T[cluster_mask]
        else:
            data_cluster_avg_masked = data_cluster_avg.reshape(data_cluster_avg.shape[0], data_cluster_avg.shape[1]*data_cluster_avg.shape[2]).T[mask][cluster_mask]
        weights_masked = weights[cluster_mask]
        cluster_avg = calculate_weighted_average(data_cluster_avg_masked, weights_masked, batch_size)
        clusters_av_dataframe[var + '_cluster' + str(c+1)] = cluster_avg

    clusters_av_dataframe.index = total_data.time.values

    # Create a dataframe with the cluster labels
    labels_dataframe = pd.DataFrame(cluster.labels, columns=['cluster'])
    labels_dataframe['nodes_lat'] = np.array(nodes_list)[:,0]
    labels_dataframe['nodes_lon'] = np.array(nodes_list)[:,1]
    labels_dataframe['cluster'] = labels_dataframe['cluster'] + 1

    return centroids, centroids_dataframe, clusters_av_dataframe, labels_dataframe
