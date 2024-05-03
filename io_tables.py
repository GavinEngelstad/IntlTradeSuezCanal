import geopandas as gpd
import networkx as nx
import pandas as pd
import numpy as np


def get_io_table(file, n_country, n_industry):
    ## setup table
    io_table = pd.read_csv(
        file, # read the file
    )
    # indexs
    io_table.iloc[n_country*n_industry:, 0] = 'TOT_' + io_table.iloc[n_country*n_industry:, 0]
    io_table['country'] = io_table['V1'].str[:3]
    io_table['industry'] = io_table['V1'].str[4:]
    io_table = io_table.set_index(
            ['country', 'industry']
        ).drop(
            columns='V1'
        ).rename_axis(
            [None, None]
        )

    # headers
    io_table = io_table.rename(columns={'OUT': 'TOT_OUT'})
    io_table.columns = pd.MultiIndex.from_arrays([io_table.columns.str[:3], io_table.columns.str[4:]])

    return io_table


def get_matricies(io_table,
                  n_country,
                  n_industry,
                  n_final_use,
                  n_value_added):
    ## Make the basic matricies
    Z = io_table.iloc[0:n_country*n_industry, 0:n_country*n_industry] # output matrix
    F = io_table.iloc[0:n_country*n_industry, n_country*n_industry:n_country*(n_industry+n_final_use)] # final use matrix
    W = io_table.iloc[n_country*n_industry+1:n_country*n_industry+n_value_added, 0:n_country*n_industry].transpose() # value added matrix
    X_c = io_table.iloc[0:n_country*n_industry, n_country*(n_industry+n_final_use)] # final output (from use)
    X_r = io_table.iloc[n_country*n_industry+n_value_added, 0:n_country*n_industry] # final output (from production)
    assert (X_c == X_r).all()
    X = X_c # set total output
    A = Z.div(X).replace(np.NAN, 0) # output per unit of production (Leontiff Matrix)
    V  = W.div(X, axis=0).replace(np.NAN, 0) # value added per unit of production
    L_inv = pd.DataFrame(np.linalg.inv(np.identity(A.shape[0]) - A), A.columns, A.index) # Leontiff Inverse

    return Z, F, W, X, A, V, L_inv


def check_matricies(Z, F, W, X, A, V, L_inv, tol):
    ## Check matricies
    assert (np.abs(Z @ np.ones(Z.shape[1]) + F @ np.ones(F.shape[1]) - X) < tol).all() # output accross rows is equal (2)
    assert (np.abs(A @ X - Z @ np.ones(Z.shape[1])) < tol).all() # check leontiff matrix and z identity work (4)
    assert (np.abs(L_inv @ F @ np.ones(F.shape[1]) - X) < tol).all() # Leontiff equation (5)
    assert (np.abs(np.array(np.diag((V @ np.ones(V.shape[1]))) @ L_inv @ (F @ np.ones(F.shape[1]))) - W @ np.ones(W.shape[1])) < tol).all() # value added identity (6)


def get_value_chain_adj(F, A, V, L_inv):
    return (np.diag((V @ np.ones(V.shape[1]))) @ L_inv @ F).set_axis(A.index).groupby(level=0).sum().groupby(level=0, axis=1).sum() # los paper


def get_node_reliance(value_adj,
                      node='maritime2927',
                      key='import',
                      port_file='data/Global port supply-chains/Port_statistics/port_locations_value.csv',
                      edge_file='data/Global port supply-chains/Network/edges_maritime_corrected.gpkg'):
    # ports (to weight)
    ports = pd.read_csv(port_file)
    ports = ports[['id', 'name', 'iso3', 'geometry', 'lat', 'lon', 'export', 'import', 'throughput']]

    # network (to get shortest path)
    edges = gpd.read_file(edge_file)
    G = nx.from_pandas_edgelist(
        edges,
        source='from_id',
        target='to_id',
        edge_attr='distance'
    )

    # set it all to 0s 
    suez_reliance = value_adj.copy()
    suez_reliance.loc[:, :] = 0

    for c1 in suez_reliance.index: # for each country
        c1_ports = ports[ports['iso3'] == c1]
        c1_total = c1_ports[key].sum()
        if c1_total == 0: # nan if not on the water
            suez_reliance.loc[c1, :] = np.nan
            suez_reliance.loc[:, c1] = np.nan
        for p1 in c1_ports.index: # for each port in c1
            sps = nx.shortest_path( # shortest path to everywhere
                G,
                source=ports['id'][p1],
                weight='distance'
            )
            for c2 in suez_reliance.loc[c1:].index: # for each country
                c2_ports = ports[ports['iso3'] == c2]
                c2_total = c2_ports[key].sum()
                suez_count = suez_reliance.loc[c1, c2]
                for p2 in c2_ports.index: # for port in c2
                    suez_count += (node in sps[ports['id'][p2]]) * ports[key][p1]*ports[key][p2] / (c1_total*c2_total) # weight by port value over total country value
                suez_reliance.loc[c1, c2] = suez_count
                suez_reliance.loc[c2, c1] = suez_count

    return suez_reliance


def get_country_stats(value_adj, suez_reliance):
    country_stats = pd.DataFrame(index=suez_reliance.index)
    country_stats['v_total'] = value_adj.sum()
    country_stats['v_through_suez'] = (suez_reliance * value_adj).sum().replace(0, np.nan) # 0s -> nan+nan+nan+..., there are no meaningful 0s
    country_stats['pct_v_through_suez'] = 100*country_stats['v_through_suez'] / country_stats['v_total']
    country_stats['v_blocked'] = country_stats['v_through_suez'] * 6/365
    country_stats['pct_v_blocked'] = country_stats['pct_v_through_suez'] * 6/365
    return country_stats
