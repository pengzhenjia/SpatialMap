import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

def get_KNN_edge_index(pos,Kn=3):

    nbrs = NearestNeighbors(n_neighbors=Kn+1, algorithm='auto').fit(pos)
    distances, indices = nbrs.kneighbors(pos)
    edge_list = np.array([[i, indices[i, j]] for i in range(len(pos)) for j in range(1, Kn+1)])

    return edge_list


def load_data(scfilename,srtfilename,Kn=3):

    sc_adata=sc.read_h5ad(scfilename)
    srt_adata=sc.read_h5ad(srtfilename)

    cell_type=sorted(list(set(sc_adata.obs['label'])))

    sc_x=sc_adata.X
    sc_y= list(sc_adata.obs['label'])
    srt_x= srt_adata.X
    srt_p=srt_adata.obs[['x','y']]

    cell_type_dict = {}
    inverse_dict = {}    
    for i, cell_type in enumerate(cell_type):
        cell_type_dict[cell_type] = i
        inverse_dict[i] = cell_type
    sc_y = np.array([cell_type_dict[x] for x in sc_y])
    
    srt_edges = get_KNN_edge_index(srt_p, Kn)
    undirected_edges = []
    for edge in srt_edges:
        u, v = edge
        undirected_edges.append([u, v])
        undirected_edges.append([v, u])
    srt_edges=undirected_edges
    
#     print(count_neighbor(srt_edges))

    #preprocess
    #replace nan
    srt_x = np.nan_to_num(srt_x, nan=0)
    sc_x = np.nan_to_num(sc_x, nan=0)
    #normalize
    epsilon = 1e-8
    means = np.mean(sc_x, axis=1, keepdims=True)
    stds = np.std(sc_x, axis=1, keepdims=True)
    stds[stds == 0] = epsilon
    sc_x = (sc_x - means) / stds

    means = np.mean(srt_x, axis=1, keepdims=True)
    stds = np.std(srt_x, axis=1, keepdims=True)
    stds[stds == 0] = epsilon
    srt_x = (srt_x - means) / stds
  
    return sc_x, sc_y, srt_x, srt_edges, cell_type_dict, inverse_dict
    

def load_novel_data(novel, scfilename,srtfilename, scsample_rate,srtsample_rate, Kn=3):

    sc=scanpy.read_h5ad(scfilename)
    srt=scanpy.read_h5ad(srtfilename)

    gene_expression = pd.DataFrame(srt.X, columns=srt.var.index, index=srt.obs.index)
    cell_info = srt.obs[['x', 'y', 'label']]
    combined_df = pd.concat([gene_expression, cell_info], axis=1)
    combined_df.columns = list(srt.var.index) + ['x', 'y', 'label']
    srt=combined_df

    gene_expression = pd.DataFrame(sc.X, columns=sc.var.index, index=sc.obs.index)
    cell_info = sc.obs[['label']]
    combined_df = pd.concat([gene_expression, cell_info], axis=1)
    combined_df.columns = list(sc.var.index) + ['label']
    sc=combined_df

    common_cell_type=sorted(list(set(sc['label'])&set(srt['label'])))+novel
    sc = sc[sc['label'].isin(common_cell_type)]
    srt = srt[srt['label'].isin(common_cell_type)]
    print(sc.shape,srt.shape,len(common_cell_type))

    if scsample_rate:
        sc=sc.sample(n=round(scsample_rate*len(sc)), random_state=1)
    if srtsample_rate:
        srt['original_index'] = list(range(len(srt)))
        srt=srt.sample(n=round(srtsample_rate*len(srt)), random_state=1)
        srt_index=srt['original_index'].to_list()
        srt = srt.drop(columns=['original_index'])
    print(sc.shape,srt.shape,len(common_cell_type))

    sc_x=sc.iloc[:, :-1].values
    sc_y= sc['label'].to_list()
    srt_x=srt.iloc[:, :-3].values
    srt_p=srt.iloc[:, -3:-1].values
    srt_y= srt['label'].to_list()


    cell_type_dict = {}
    inverse_dict = {}    
    for i, cell_type in enumerate(common_cell_type):
        cell_type_dict[cell_type] = i
        inverse_dict[i] = cell_type
    sc_y = np.array([cell_type_dict[x] for x in sc_y])
    srt_y = np.array([cell_type_dict[x] for x in srt_y])

    # sc_edges = get_tonsilbe_edge_index(sc_x, scdistance_thres, source_dist, 'sc')
    srt_edges = get_KNN_edge_index(srt_p, Kn)
    # print(count_neighbor(sc_edges))
    print(count_neighbor(srt_edges))


    # 预处理
    missing_values_count = np.isnan(srt_x).sum()
    print(f"srt 中存在 {missing_values_count} 个缺失值")
    missing_values_count = np.isnan(sc_x).sum()
    print(f"sc 中存在 {missing_values_count} 个缺失值")
    # 填充缺失值，例如填充为 0
    srt_x = np.nan_to_num(srt_x, nan=0)
    sc_x = np.nan_to_num(sc_x, nan=0)
    # print("已将缺失值填充为 0")

    epsilon = 1e-8  # 防止除以 0 的小值
    means = np.mean(sc_x, axis=1, keepdims=True)
    stds = np.std(sc_x, axis=1, keepdims=True)
    stds[stds == 0] = epsilon
    sc_x = (sc_x - means) / stds

    means = np.mean(srt_x, axis=1, keepdims=True)
    stds = np.std(srt_x, axis=1, keepdims=True)
    stds[stds == 0] = epsilon
    srt_x = (srt_x - means) / stds
    
    if srtsample_rate:
        return sc_x, sc_y, srt_x, srt_y, srt_edges, cell_type_dict, inverse_dict, srt_index
    else:
        return sc_x, sc_y, srt_x, srt_y, srt_edges, cell_type_dict, inverse_dict