import geopandas as gpd
import networkx as nx
import pandas as pd
import numpy as np
import shapely


'''
Callable functions to make the shipping network and find canal reiance values
'''


def geodataframe_from_csv(file: str) -> gpd.GeoDataFrame:
    '''
    Take CSV with a 'geometry' column and returns a GeoDataFrame
    '''
    df = pd.read_csv(file) # lead file
    df['geometry'] = df['geometry'].apply(shapely.wkt.loads) # setup geometry
    return gpd.GeoDataFrame(df, crs='epsg:4326') # make geodataframe


def combine_overlapping_edges(df: pd.DataFrame,
                              merge_on: list[str],
                              sum_keys: list[str],
                              attr_keys: list[str]
                              ) -> pd.DataFrame:
    '''
    Takes a dataframe edge list, keys to merge on, keys to sum, and keys to keep
    the first of and returns a dataframe edge list which combines any edges that
    have the same start and end (in any order) with the sum_keys added and attr_keys
    kept.
    '''
    # sumed bits
    grouped_sum = df[merge_on + sum_keys].groupby(
        df[merge_on].apply(
                lambda row: '|'.join(sorted(row)),
                axis='columns'
            )
        ).sum().reset_index(drop=False)
    grouped_sum[merge_on] = grouped_sum['index'].str.split('|', expand=True)
    grouped_sum.drop(columns='index', inplace=True)

    # get first bits
    grouped_first = df[merge_on + attr_keys].groupby(
        df[merge_on].apply(
                lambda row: '|'.join(sorted(row)),
                axis='columns'
            )
        ).first().reset_index(drop=False)
    grouped_first[merge_on] = grouped_first['index'].str.split('|', expand=True)
    grouped_first.drop(columns='index', inplace=True)

    # merge
    grouped = grouped_first.merge(
            right=grouped_sum,
            on=merge_on,
            how='outer'
        )
    return grouped


def add_earth(ax):
    '''
    Adds a blank, gray earth to a map
    '''
    gpd.read_file(gpd.datasets.get_path("naturalearth_lowres")).plot(alpha=0.5, color="gray", ax=ax)


def canal_reliance(shipping_network: pd.DataFrame,
                         shipping_network_through_canal: pd.DataFrame,
                         canal_name: str = 'canal'
                         ) -> pd.DataFrame:
    '''
    Takes a edgelist for the network and an edgelist for the network through a canal
    and returns a merges list including canal reliance scores
    '''
    # remove unimportant data
    shipping_network_through_canal = shipping_network_through_canal[['from_id', 'to_id', 'v_sea_flow', 'q_sea_flow']]
    # filter to not mess names up after merge
    shipping_network_through_canal.rename(
            columns={
                    'v_sea_flow': 'v_sea_flow_' + canal_name,
                    'q_sea_flow': 'q_sea_flow_' + canal_name,
                },
            inplace=True
        )

    # merge
    shipping_network = shipping_network.merge(
            right=shipping_network_through_canal,
            on=['from_id', 'to_id'],
            how='left'
        ).replace(np.NaN, 0)

    # reliance ratios
    shipping_network['v_ratio_' + canal_name] = shipping_network['v_sea_flow_' + canal_name] / shipping_network['v_sea_flow']
    shipping_network['q_ratio_' + canal_name] = shipping_network['q_sea_flow_' + canal_name] / shipping_network['q_sea_flow']

    return shipping_network


def fastest_route_from(orgin_node: str,
                       edgelist: pd.DataFrame,
                       ports: pd.DataFrame,
                       canal_name: str = 'canal'
                       ) -> pd.DataFrame:
    '''
    Takes an orgin node, edgelist, and list of ports. Returns an amended list of ports
    that includes the path and details about the path from the orgin node to the port
    '''
    # make graph
    G = nx.from_pandas_edgelist(
            edgelist,
            source='from_id',
            target='to_id',
            edge_attr=['distance', 'length', 'geometry']
        )
    shortest_paths = nx.shortest_path(
            G, 
            source=orgin_node,
            weight='distance'
        )
    
    ports[['dist_from_' + canal_name, 'len_from_' + canal_name, 'geometry_from_' + canal_name]] = 0, 0, None # add columns to ports
    for i in ports.index:
        path = shortest_paths[ports['id'][i]] # get path from orgin
        distance = 0
        length = 0
        geometry = []
        for j in range(1, len(path)):
            piece = G[path[j-1]][path[j]]
            distance += piece['distance'] # extend the path by the distnce and length
            length += piece['length']
            if shapely.length(piece['geometry']) <= 359: # if the line doesnt cross the pacific
                geometry.append(piece['geometry']) # add the line
        ports['dist_from_' + canal_name][i] = distance # add results to dictionary
        ports['len_from_' + canal_name][i] = length
        ports['geometry_from_' + canal_name][i] = shapely.MultiLineString(geometry) # make graphable object
    return ports


def add_ma(df: pd.DataFrame,
           key: str,
           window: int = 5
           ) -> pd.DataFrame:
    '''
    Adds a moving average to the dataframe
    '''
    df['ma_' + key] = df[key].rolling(
        window=window,
        center=True
    ).mean()

    return df