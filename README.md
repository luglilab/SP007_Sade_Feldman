```python
import scanpy as sc
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import colors
import seaborn as sb
from gprofiler import GProfiler

import rpy2.rinterface_lib.callbacks
import logging

import warnings

from rpy2.robjects import pandas2ri
import anndata2ri
import pandas as pd
plt.rcParams['figure.figsize']=(8,8) #rescale figures
sc.settings.verbosity = 3
sc.set_figure_params(dpi=80, dpi_save=150)
sc.logging.print_versions()
import scanpy.external as sce
import scprep
```

    /home/spuccio/miniconda3/envs/scrnaseq2/lib/python3.6/site-packages/anndata/_core/anndata.py:21: FutureWarning: pandas.core.index is deprecated and will be removed in a future version.  The public classes are available in the top-level namespace.
      from pandas.core.index import RangeIndex
    /home/spuccio/miniconda3/envs/scrnaseq2/lib/python3.6/site-packages/rpy2/robjects/pandas2ri.py:34: UserWarning: pandas >= 1.0 is not supported.
      warnings.warn('pandas >= 1.0 is not supported.')


    scanpy==1.4.7.dev48+g41563144 anndata==0.7.1 umap==0.4.1 numpy==1.15.4 scipy==1.4.1 pandas==1.0.3 scikit-learn==0.22.1 statsmodels==0.11.0 python-igraph==0.7.1 louvain==0.6.1


# Import Raw Data


```python
adata=sc.read("/home/spuccio/datadisk2/SP007_RNA_Seq_Overexpression_MEOX1/scrnaseqmelanoma/Melanoma.txt",delimiter='\t',cache=True,first_column_names=True).transpose()
```

    ... reading from cache file cache/home-spuccio-datadisk2-SP007_RNA_Seq_Overexpression_MEOX1-scrnaseqmelanoma-Melanoma.h5ad



```python
adata
```




    AnnData object with n_obs × n_vars = 16291 × 55737 



# Import Metadata


```python
Metadata = pd.read_csv("/home/spuccio/datadisk2/SP007_RNA_Seq_Overexpression_MEOX1/scrnaseqmelanoma/MelanomaNewMeta.txt",sep="\t",header=None)
```


```python
Metadata.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A10_P3_M11</td>
      <td>Sample-1</td>
      <td>Melanoma</td>
      <td>Biopsy</td>
      <td>SmartSeq2</td>
      <td>Pre</td>
      <td>P1</td>
      <td>Responder</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A11_P1_M11</td>
      <td>Sample-2</td>
      <td>Melanoma</td>
      <td>Biopsy</td>
      <td>SmartSeq2</td>
      <td>Pre</td>
      <td>P1</td>
      <td>Responder</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A11_P3_M11</td>
      <td>Sample-3</td>
      <td>Melanoma</td>
      <td>Biopsy</td>
      <td>SmartSeq2</td>
      <td>Pre</td>
      <td>P1</td>
      <td>Responder</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A11_P4_M11</td>
      <td>Sample-4</td>
      <td>Melanoma</td>
      <td>Biopsy</td>
      <td>SmartSeq2</td>
      <td>Pre</td>
      <td>P1</td>
      <td>Responder</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A12_P3_M11</td>
      <td>Sample-5</td>
      <td>Melanoma</td>
      <td>Biopsy</td>
      <td>SmartSeq2</td>
      <td>Pre</td>
      <td>P1</td>
      <td>Responder</td>
    </tr>
  </tbody>
</table>
</div>




```python
adata.obs['CellID'] = Metadata[1].to_list()
adata.obs['SampleID'] = Metadata[6].to_list()
adata.obs['Therapy'] = Metadata[5].to_list()
adata.obs['Outcome'] = Metadata[7].to_list()

```


```python
adata
```




    AnnData object with n_obs × n_vars = 16291 × 55737 
        obs: 'CellID', 'SampleID', 'Therapy', 'Outcome'



# Quality control


```python
scprep.plot.plot_library_size(adata[:, ].to_df()>0,log=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f054ab63b00>




![png](output_10_1.png)



```python
scprep.plot.plot_gene_set_expression(adata[:, ].to_df())
```

    /home/spuccio/.local/lib/python3.6/site-packages/scprep/select.py:362: UserWarning: No selection conditions provided. Returning all columns.
      "No selection conditions provided. " "Returning all columns.", UserWarning





    <matplotlib.axes._subplots.AxesSubplot at 0x7f05148b1748>




![png](output_11_2.png)



```python
#sc.pp.log1p(adata)
```


```python
adata.var_names_make_unique() 
```


```python
# Quality control - calculate QC covariates
adata.obs['n_counts'] = adata.X.sum(1)
adata.obs['log_counts'] = np.log(adata.obs['n_counts'])
adata.obs['n_genes'] = (adata.X > 0).sum(1)
```


```python
mito_genes = adata.var_names.str.startswith('MT-')
ribo_genes = adata.var_names.str.startswith(("RPS","RPL"))
print(sum(mito_genes))
print(sum(ribo_genes))
```

    37
    927



```python
adata.obs['mt_frac'] = np.sum(
    adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
```


```python
adata.obs['ribo_frac'] = np.sum(
    adata[:, ribo_genes].X, axis=1) / np.sum(adata.X, axis=1)
```


```python
sc.pl.violin(adata, ['n_counts', 'n_genes','mt_frac','ribo_frac'],
             jitter=0.4, groupby = 'Therapy',save='QC_before.png')
```

    ... storing 'SampleID' as categorical
    ... storing 'Therapy' as categorical
    ... storing 'Outcome' as categorical


    WARNING: saving figure to file figures/violinQC_before.png



![png](output_18_2.png)



```python
print(adata.obs['Therapy'].value_counts())
```

    Post    10363
    Pre      5928
    Name: Therapy, dtype: int64



```python
print(adata.obs['Outcome'].value_counts())
```

    Non-responder    10727
    Responder         5564
    Name: Outcome, dtype: int64



```python
cc_genes_file ="/home/spuccio/datadisk2/SP007_RNA_Seq_Overexpression_MEOX1/MelanomaDurante/single-cell-tutorial/regev_lab_cell_cycle_genes.txt"
```

# Preprocessing


```python
sc.pl.scatter(adata, x='n_counts', y='mt_frac')
sc.pl.scatter(adata, x='n_counts', y='n_genes')
```


![png](output_23_0.png)



![png](output_23_1.png)



```python
scprep.plot.plot_gene_set_expression(adata[:, ].to_df())
```

    /home/spuccio/.local/lib/python3.6/site-packages/scprep/select.py:362: UserWarning: No selection conditions provided. Returning all columns.
      "No selection conditions provided. " "Returning all columns.", UserWarning





    <matplotlib.axes._subplots.AxesSubplot at 0x7f0548eda6a0>




![png](output_24_2.png)



```python
sc.pp.log1p(adata)
```


```python
adata.raw = adata
```


```python
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
```

    extracting highly variable genes
        finished (0:00:11)
    --> added
        'highly_variable', boolean vector (adata.var)
        'means', float vector (adata.var)
        'dispersions', float vector (adata.var)
        'dispersions_norm', float vector (adata.var)



```python
sc.pl.highly_variable_genes(adata)
```


![png](output_28_0.png)



```python
sc.pp.regress_out(adata[:, adata[:,].to_df().sum(axis=0) > 0], ['n_counts', 'mt_frac'])
```

    regressing out ['n_counts', 'mt_frac']
        finished (0:12:33)


# Principal component analysis


```python
sc.tl.pca(adata,n_comps=50, svd_solver='arpack')
```

        on highly variable genes
    computing PCA with n_comps = 50
        finished (0:00:06)



```python
sc.pl.pca(adata, color=['Therapy','Outcome'])
```


![png](output_32_0.png)



```python
sc.pl.pca(adata, color='n_counts')
```


![png](output_33_0.png)



```python
sc.pl.pca_variance_ratio(adata, log=True)
```


![png](output_34_0.png)



```python
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
```

    computing neighbors
        using 'X_pca' with n_pcs = 40
        finished: added to `.uns['neighbors']`
        `.obsp['distances']`, distances for each pair of neighbors
        `.obsp['connectivities']`, weighted adjacency matrix (0:00:09)


# Umap


```python
sc.tl.umap(adata,n_components=3,random_state=10)
```

    computing UMAP
        finished: added
        'X_umap', UMAP coordinates (adata.obsm) (0:00:14)



```python
sc.pl.umap(adata, color=['CD3E', 'CD8A', 'CD4'])
```


![png](output_38_0.png)



```python
fig, (ax1, ax2,ax3) = plt.subplots(1,3, figsize=(33/2.54, 17/2.54))

scprep.plot.scatter(x=adata[:, ['CD3E']].to_df(), y=adata[:, ['CD8A']].to_df(), c=adata[:, ['CD8A']].to_df(),  ax=ax1,
                    xlabel='CD3E Normalized Count Log2(TPM)', ylabel='CD8A Normalized Count Log2(TPM)', legend_title="CD8A", title='CD8A+ Cells')

scprep.plot.scatter(x=adata[:, ['CD3E']].to_df(), y=adata[:, ['CD4']].to_df(), c=adata[:, ['CD4']].to_df(),  ax=ax2,
                    xlabel='CD3E Normalized Count Log2(TPM)', ylabel='CD3E Normalized Count Log2(TPM)', legend_title="CD8A", title='CD4+ Cells')
scprep.plot.plot_gene_set_expression(adata[:, ['CD8A']].to_df().loc[adata[:, ['CD8A']].to_df()['CD8A']>=0.001],xlabel='CD8A Normalized Count Log2(TPM)',cutoff=2,ax=ax3,title="CD8A+ Threshold")

plt.tight_layout()
plt.show()
```

    /home/spuccio/.local/lib/python3.6/site-packages/scprep/select.py:362: UserWarning: No selection conditions provided. Returning all columns.
      "No selection conditions provided. " "Returning all columns.", UserWarning



![png](output_39_1.png)


# Magic Imputation on CD45+


```python
adata_magic = adata.copy()
import scanpy.external as sce
#sce.pp.magic(adata_magic[:, adata_magic[:,].to_df().sum(axis=0) > 0.01], name_list="all_genes", knn=5)
sce.pp.magic(adata_magic, name_list="all_genes", knn=10)
```

    computing MAGIC


    /home/spuccio/.local/lib/python3.6/site-packages/magic/magic.py:472: UserWarning: Input matrix contains unexpressed genes. Please remove them prior to running MAGIC.
      "Input matrix contains unexpressed genes. "


      Running MAGIC with `solver='exact'` on 55737-dimensional data may take a long time. Consider denoising specific genes with `genes=<list-like>` or using `solver='approximate'`.
        finished (0:02:21)



```python
colors2 = plt.cm.Reds(np.linspace(0, 1, 128))
colors3 = plt.cm.Greys_r(np.linspace(0.7,0.8,20))
colorsComb = np.vstack([colors3, colors2])
mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colorsComb)
```

# UMAP plot of CD45+ Before Imputation plotting some marker genes


```python
#sc.pp.scale(adata_fil)
sc.pl.umap(adata, color=['IL7R', 'CD8A', 'CD8B','CD4',  
                'LGALS3',  'BAG3', 'IFNG', 'PRDM1',
                'MEOX1','TRPC1','MAF','SOX7'], use_raw=False, color_map=mymap)
```


![png](output_44_0.png)


# UMAP plot of CD45+ After Imputation plotting some marker genes


```python
#sc.pp.scale(adata_fil)
sc.pl.umap(adata_magic, color=['IL7R', 'CD8A', 'CD8B','CD4',  
                'LGALS3',  'BAG3', 'IFNG', 'PRDM1',
                'MEOX1','TRPC1','MAF','SOX7'], use_raw=False, color_map=mymap)
```


![png](output_46_0.png)


# MEOX1 Expression Before/After Imputation 


```python
#fig, (ax1, ax2,ax3) = plt.subplots(1,3, figsize=(33/2.54, 17/2.54))
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(33/2.54, 17/2.54))
scprep.plot.scatter(x=adata[:, ['CD3E']].to_df(), y=adata[:, ['CD8A']].to_df(), c=adata[:, ['MEOX1']].to_df(),  ax=ax1,cmap=mymap,
                    xlabel='CD3E Normalized Count Log2(TPM)', ylabel='CD8A Normalized Count Log2(TPM)', legend_title="MEOX1", title='CD45+ Before MAGIC')


scprep.plot.scatter(x=adata_magic[:, ['CD3E']].to_df(), y=adata_magic[:, ['CD8A']].to_df(), c=adata_magic[:, ['MEOX1']].to_df(),cmap=mymap,  ax=ax2,
                    xlabel='CD3E Normalized Log2(TPM) (Imputed) ', ylabel='CD8A Normalized Count  Log2(TPM) (Imputed) ', legend_title="MEOX1", title='CD45+ After MAGIC')

plt.tight_layout()
plt.show()
#plt.savefig("/mnt/hpcserver1_datadisk2_spuccio/SP007_RNA_Seq_Overexpression_MEOX1/MelanomaDurante/figures/MAGIC.png")
```


![png](output_48_0.png)


# Filter CD8+ Cells using >=2 Log2 TPM  as threshold


```python
adata_sub_CD8_POS = adata[(adata[:,'CD8A'].X>=2).flatten(), : ] 
```

## Guide lines suggest to start from row data and perform analysis step applied only to the selected cells


```python
adata_sub_CD8_POS
```




    View of AnnData object with n_obs × n_vars = 6423 × 55737 
        obs: 'CellID', 'SampleID', 'Therapy', 'Outcome', 'n_counts', 'log_counts', 'n_genes', 'mt_frac', 'ribo_frac'
        var: 'highly_variable', 'means', 'dispersions', 'dispersions_norm'
        uns: 'Therapy_colors', 'log1p', 'pca', 'Outcome_colors', 'neighbors', 'umap'
        obsm: 'X_pca', 'X_umap'
        varm: 'PCs'
        obsp: 'distances', 'connectivities'




```python
adata=sc.read("/home/spuccio/datadisk2/SP007_RNA_Seq_Overexpression_MEOX1/scrnaseqmelanoma/Melanoma.txt",delimiter='\t',cache=True,first_column_names=True).transpose()
```

    ... reading from cache file cache/home-spuccio-datadisk2-SP007_RNA_Seq_Overexpression_MEOX1-scrnaseqmelanoma-Melanoma.h5ad



```python
adata.obs['CellID'] = Metadata[1].to_list()
adata.obs['SampleID'] = Metadata[6].to_list()
adata.obs['Therapy'] = Metadata[5].to_list()
adata.obs['Outcome'] = Metadata[7].to_list()
```


```python
adata_CD8 =  adata[adata_sub_CD8_POS.obs_names].copy()
```


```python
# Quality control - calculate QC covariates
adata_CD8.obs['n_counts'] = adata_CD8.X.sum(1)
adata_CD8.obs['log_counts'] = np.log(adata_CD8.obs['n_counts'])
adata_CD8.obs['n_genes'] = (adata_CD8.X > 0).sum(1)
```


```python
adata_CD8.obs['mt_frac'] = np.sum(
    adata_CD8[:, mito_genes].X, axis=1) / np.sum(adata_CD8.X, axis=1)
```


```python
adata_CD8.obs['ribo_frac'] = np.sum(
    adata_CD8[:, ribo_genes].X, axis=1) / np.sum(adata_CD8.X, axis=1)
```


```python
sc.pl.violin(adata_CD8, ['n_counts', 'n_genes','mt_frac','ribo_frac'],
             jitter=0.4, groupby = 'Therapy')
```

    ... storing 'SampleID' as categorical
    ... storing 'Therapy' as categorical
    ... storing 'Outcome' as categorical



![png](output_59_1.png)



```python
print(adata_CD8.obs['Therapy'].value_counts())
```

    Post    3937
    Pre     2486
    Name: Therapy, dtype: int64



```python
print(adata_CD8.obs['Outcome'].value_counts())
```

    Non-responder    4459
    Responder        1964
    Name: Outcome, dtype: int64



```python
sc.pl.scatter(adata_CD8, x='n_counts', y='mt_frac')
sc.pl.scatter(adata_CD8, x='n_counts', y='n_genes')
```


![png](output_62_0.png)



![png](output_62_1.png)



```python
sc.pp.log1p(adata_CD8)
```


```python
sc.pp.highly_variable_genes(adata_CD8, min_mean=0.0125, max_mean=3, min_disp=0.5)
```

    extracting highly variable genes
        finished (0:00:06)
    --> added
        'highly_variable', boolean vector (adata.var)
        'means', float vector (adata.var)
        'dispersions', float vector (adata.var)
        'dispersions_norm', float vector (adata.var)



```python
sc.pl.highly_variable_genes(adata_CD8)
```


![png](output_65_0.png)



```python
sc.pp.regress_out(adata_CD8[:, adata_CD8[:,].to_df().sum(axis=0) > 0], ['n_counts', 'mt_frac'])
```

    regressing out ['n_counts', 'mt_frac']
        finished (0:05:48)



```python
sc.tl.pca(adata_CD8,n_comps=50, svd_solver='arpack')
```

        on highly variable genes
    computing PCA with n_comps = 50
        finished (0:00:04)



```python
sc.pl.pca(adata_CD8, color=['Therapy','Outcome'])
```


![png](output_68_0.png)



```python
sc.pp.neighbors(adata_CD8, n_neighbors=10, n_pcs=40)
```

    computing neighbors
        using 'X_pca' with n_pcs = 40
        finished: added to `.uns['neighbors']`
        `.obsp['distances']`, distances for each pair of neighbors
        `.obsp['connectivities']`, weighted adjacency matrix (0:00:01)



```python
sc.tl.umap(adata_CD8,n_components=3,random_state=10)
```

    computing UMAP
        finished: added
        'X_umap', UMAP coordinates (adata.obsm) (0:00:14)



```python
sc.pl.umap(adata_CD8, color=['CD3E', 'CD8A', 'CD4'])
```


![png](output_71_0.png)



```python
sc.pl.umap(adata_CD8, color=['Outcome','Therapy','SampleID'])
```


![png](output_72_0.png)


# Imputation of CD8+ Cells


```python
adata_magic_CD8 = adata_CD8.copy()
import scanpy.external as sce
#sce.pp.magic(adata_magic[:, adata_magic[:,].to_df().sum(axis=0) > 0.01], name_list="all_genes", knn=5)
sce.pp.magic(adata_magic_CD8, name_list="all_genes", knn=5)
```

    computing MAGIC


    /home/spuccio/.local/lib/python3.6/site-packages/magic/magic.py:472: UserWarning: Input matrix contains unexpressed genes. Please remove them prior to running MAGIC.
      "Input matrix contains unexpressed genes. "


      Running MAGIC with `solver='exact'` on 55737-dimensional data may take a long time. Consider denoising specific genes with `genes=<list-like>` or using `solver='approximate'`.
        finished (0:00:29)


# UMAP plot of CD8+ Before Imputation plotting some marker genes


```python
#sc.pp.scale(adata_fil)
sc.pl.umap(adata_CD8, color=['IL7R', 'CD8A', 'CD8B','CD4',  
                'FOXP3',  'GNLY', 'NKG7', 'KLRB1',
                'MEOX1','PRF1','GZMK','TOX'], use_raw=False, color_map=mymap)
```


![png](output_76_0.png)


# UMAP plot of CD8+ After Imputation plotting some marker genes


```python
#sc.pp.scale(adata_fil)
sc.pl.umap(adata_magic_CD8, color=['IL7R', 'CD8A', 'CD8B','CD4',  
                'FOXP3',  'GNLY', 'NKG7', 'KLRB1',
                'MEOX1','PRF1','GZMK','TOX'], use_raw=False, color_map=mymap)
```


![png](output_78_0.png)


# MEOX1 Expression Before/After Imputation 


```python
#fig, (ax1, ax2,ax3) = plt.subplots(1,3, figsize=(33/2.54, 17/2.54))
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(33/2.54, 17/2.54))
scprep.plot.scatter(x=adata_CD8[:, ['CD3E']].to_df(), y=adata_CD8[:, ['CD8A']].to_df(), c=adata_CD8[:, ['MEOX1']].to_df(),  ax=ax1,cmap=mymap,
                    xlabel='CD3E Normalized Log2(TPM)', ylabel='CD8A Normalized Count Log2(TPM)', legend_title="MEOX1", title='CD8+ Before MAGIC')


scprep.plot.scatter(x=adata_magic_CD8[:, ['CD3E']].to_df(), y=adata_magic_CD8[:, ['CD8A']].to_df(), c=adata_magic_CD8[:, ['MEOX1']].to_df(),cmap=mymap,  ax=ax2,
                    xlabel='CD3E Normalized Log2(TPM) (Imputed)', ylabel='CD8A Normalized Count  Log2(TPM) (Imputed) ', legend_title="MEOX1", title='CD8+ After MAGIC')

plt.tight_layout()
plt.show()
```


![png](output_80_0.png)


# Analysis Metrics


```python
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(33/2.54, 17/2.54))
adata.obs['SampleID'].value_counts().sort_index().plot.bar(x=adata.obs['SampleID'].value_counts().sort_index().index, y=adata.obs['SampleID'].value_counts().sort_index(),color=adata_CD8.uns['SampleID_colors'],ax = ax1)
adata_CD8.obs['SampleID'].value_counts().sort_index().plot.bar(x=adata_CD8.obs['SampleID'].value_counts().sort_index().index, y=adata_CD8.obs['SampleID'].value_counts().sort_index(),color=adata_CD8.uns['SampleID_colors'],ax = ax2)
ax1.set_ylabel("SampleID Cell count")
ax1.set_title("TME Cells")
ax2.set_title("CD8+ Cells")


plt.tight_layout()
plt.show()
```


![png](output_82_0.png)



```python
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(33/2.54, 17/2.54))
adata.obs['Therapy'].value_counts().sort_index().plot.bar(x=adata.obs['Therapy'].value_counts().sort_index().index, y=adata.obs['Therapy'].value_counts().sort_index(),ax = ax1)
adata_CD8.obs['Therapy'].value_counts().sort_index().plot.bar(x=adata_CD8.obs['Therapy'].value_counts().sort_index().index, y=adata_CD8.obs['Therapy'].value_counts().sort_index(),ax = ax2)
ax1.set_ylabel("Therapy Cell count")
ax1.set_title("TME Cells")
ax2.set_title("CD8+ Cells")

plt.tight_layout()
plt.show()
```


![png](output_83_0.png)



```python
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(33/2.54, 17/2.54))
adata.obs['Outcome'].value_counts().sort_index().plot.bar(x=adata.obs['Outcome'].value_counts().sort_index().index, y=adata.obs['Outcome'].value_counts().sort_index(),rot=90,ax = ax1)
adata_CD8.obs['Outcome'].value_counts().sort_index().plot.bar(x=adata_CD8.obs['Outcome'].value_counts().sort_index().index, y=adata_CD8.obs['Outcome'].value_counts().sort_index(),rot=90,ax = ax2)
ax1.set_ylabel("Outcome  Cell count")
ax1.set_title("TME Cells")
ax2.set_title("CD8+ Cells")
plt.tight_layout()
plt.show()
```


![png](output_84_0.png)



```python
#fig, (ax1, ax2,ax3) = plt.subplots(1,3, figsize=(33/2.54, 17/2.54))
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(33/2.54, 17/2.54))
scprep.plot.scatter(x=adata_sub_CD8_POS[:, ['CD3E']].to_df(), y=adata_sub_CD8_POS[:, ['CD8A']].to_df(), c=adata_sub_CD8_POS[:, ['MEOX1']].to_df(),  ax=ax1,cmap=mymap,
                    xlabel='CD3E Normalized Count (CPM)', ylabel='CD8A Normalized Count  (CPM)', legend_title="MEOX1", title='Before MAGIC')


scprep.plot.scatter(x=adata_magic_CD8[:, ['CD3E']].to_df(), y=adata_magic_CD8[:, ['CD8A']].to_df(), c=adata_magic_CD8[:, ['MEOX1']].to_df(),cmap=mymap,  ax=ax2,
                    xlabel='CD3E Normalized Count (CPM)', ylabel='CD8A Normalized Count  (CPM)', legend_title="MEOX1", title='After MAGIC')

plt.tight_layout()
plt.show()
#plt.savefig("/mnt/hpcserver1_datadisk2_spuccio/SP007_RNA_Seq_Overexpression_MEOX1/MelanomaDurante/figures/MAGIC.png")
```


![png](output_85_0.png)


# Keep cells with CD8A>2.3


```python
adata_sub_CD8_POS = adata_magic_CD8[(adata_magic_CD8[:,'CD8A'].X>=2.3).flatten(), : ] 
```


```python
adata=sc.read("/home/spuccio/datadisk2/SP007_RNA_Seq_Overexpression_MEOX1/scrnaseqmelanoma/Melanoma.txt",delimiter='\t',cache=True,first_column_names=True).transpose()
```

    ... reading from cache file cache/home-spuccio-datadisk2-SP007_RNA_Seq_Overexpression_MEOX1-scrnaseqmelanoma-Melanoma.h5ad



```python
adata.obs['CellID'] = Metadata[1].to_list()
adata.obs['SampleID'] = Metadata[6].to_list()
adata.obs['Therapy'] = Metadata[5].to_list()
adata.obs['Outcome'] = Metadata[7].to_list()
```


```python
adata_CD8_2 =  adata[adata_sub_CD8_POS.obs_names].copy()
```


```python
# Quality control - calculate QC covariates
adata_CD8_2.obs['n_counts'] = adata_CD8_2.X.sum(1)
adata_CD8_2.obs['log_counts'] = np.log(adata_CD8_2.obs['n_counts'])
adata_CD8_2.obs['n_genes'] = (adata_CD8_2.X > 0).sum(1)
```


```python
adata_CD8_2.obs['mt_frac'] = np.sum(
    adata_CD8_2[:, mito_genes].X, axis=1) / np.sum(adata_CD8_2.X, axis=1)
```


```python
adata_CD8_2.obs['ribo_frac'] = np.sum(
    adata_CD8_2[:, ribo_genes].X, axis=1) / np.sum(adata_CD8_2.X, axis=1)
```


```python
sc.pp.log1p(adata_CD8_2)
```


```python
sc.pp.highly_variable_genes(adata_CD8_2, min_mean=0.0125, max_mean=3, min_disp=0.5)
```

    extracting highly variable genes
        finished (0:00:06)
    --> added
        'highly_variable', boolean vector (adata.var)
        'means', float vector (adata.var)
        'dispersions', float vector (adata.var)
        'dispersions_norm', float vector (adata.var)



```python
sc.pp.regress_out(adata_CD8_2[:, adata_CD8_2[:,].to_df().sum(axis=0) > 0], ['n_counts', 'mt_frac'])
```

    regressing out ['n_counts', 'mt_frac']


    /home/spuccio/miniconda3/envs/scrnaseq2/lib/python3.6/site-packages/anndata/_core/anndata.py:1172: ImplicitModificationWarning: Initializing view as actual.
      "Initializing view as actual.", ImplicitModificationWarning
    Trying to set attribute `.obs` of view, copying.
    ... storing 'SampleID' as categorical
    Trying to set attribute `.obs` of view, copying.
    ... storing 'Therapy' as categorical
    Trying to set attribute `.obs` of view, copying.
    ... storing 'Outcome' as categorical


        finished (0:05:44)



```python
sc.tl.pca(adata_CD8_2,n_comps=50, svd_solver='arpack')
```

        on highly variable genes
    computing PCA with n_comps = 50
        finished (0:00:04)



```python
sc.pp.neighbors(adata_CD8_2, n_neighbors=10, n_pcs=40)
```

    computing neighbors
        using 'X_pca' with n_pcs = 40
        finished: added to `.uns['neighbors']`
        `.obsp['distances']`, distances for each pair of neighbors
        `.obsp['connectivities']`, weighted adjacency matrix (0:00:01)



```python
sc.tl.umap(adata_CD8_2,n_components=3,random_state=10)
```

    computing UMAP
        finished: added
        'X_umap', UMAP coordinates (adata.obsm) (0:00:15)



```python
sc.pl.umap(adata_CD8_2, color=['CD3E', 'CD8A', 'CD4'])
```

    ... storing 'SampleID' as categorical
    ... storing 'Therapy' as categorical
    ... storing 'Outcome' as categorical



![png](output_100_1.png)



```python
sc.pl.umap(adata_CD8_2, color=['Outcome','Therapy','SampleID'])
```


![png](output_101_0.png)


# Imputation


```python
adata_CD8_2_magic = adata_CD8_2.copy()
import scanpy.external as sce
#sce.pp.magic(adata_magic[:, adata_magic[:,].to_df().sum(axis=0) > 0.01], name_list="all_genes", knn=5)
sce.pp.magic(adata_CD8_2_magic, name_list="all_genes", knn=5)
```

    computing MAGIC


    /home/spuccio/.local/lib/python3.6/site-packages/magic/magic.py:472: UserWarning: Input matrix contains unexpressed genes. Please remove them prior to running MAGIC.
      "Input matrix contains unexpressed genes. "


      Running MAGIC with `solver='exact'` on 55737-dimensional data may take a long time. Consider denoising specific genes with `genes=<list-like>` or using `solver='approximate'`.
        finished (0:00:30)


# UMAP Plot before Imputation


```python
#sc.pp.scale(adata_fil)
sc.pl.umap(adata_CD8_2, color=['IL7R', 'CD8A', 'CD8B','CD4',  
                'FOXP3',  'GNLY', 'NKG7', 'KLRB1',
                'MEOX1','PRF1','GZMK','TOX'], use_raw=False, color_map=mymap)
```


![png](output_105_0.png)


# UMAP Plot after Imputation


```python
#sc.pp.scale(adata_fil)
sc.pl.umap(adata_CD8_2_magic, color=['IL7R', 'CD8A', 'CD8B','CD4',  
                'FOXP3',  'GNLY', 'NKG7', 'KLRB1',
                'MEOX1','PRF1','GZMK','TOX'], use_raw=False, color_map=mymap)
```


![png](output_107_0.png)


# MEOX1 before/after imputation


```python
#fig, (ax1, ax2,ax3) = plt.subplots(1,3, figsize=(33/2.54, 17/2.54))
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(33/2.54, 17/2.54))
scprep.plot.scatter(x=adata_CD8_2[:, ['CD3E']].to_df(), y=adata_CD8_2[:, ['CD8A']].to_df(), c=adata_CD8_2[:, ['MEOX1']].to_df(),  ax=ax1,cmap=mymap,
                    xlabel='CD3E Normalized Log2(TPM)', ylabel='CD8A Normalized Count Log2(TPM)', legend_title="MEOX1", title='CD8+ Before MAGIC')


scprep.plot.scatter(x=adata_CD8_2_magic[:, ['CD3E']].to_df(), y=adata_CD8_2_magic[:, ['CD8A']].to_df(), c=adata_CD8_2_magic[:, ['MEOX1']].to_df(),cmap=mymap,  ax=ax2,
                    xlabel='CD3E Normalized Log2(TPM) (Imputed)', ylabel='CD8A Normalized Count  Log2(TPM) (Imputed) ', legend_title="MEOX1", title='CD8+ After MAGIC')

plt.tight_layout()
plt.show()
```


![png](output_109_0.png)



```python
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(33/2.54, 17/2.54))
scprep.plot.scatter(x=adata_CD8_2[:, ['CD4']].to_df(), y=adata_CD8_2[:, ['CD8A']].to_df(), c=adata_CD8_2[:, ['MEOX1']].to_df(),  ax=ax1,cmap=mymap,
                    xlabel='CD4 Normalized Log2(TPM)', ylabel='CD8A Normalized Count Log2(TPM)', legend_title="MEOX1", title='CD4/CD8 Before MAGIC')


scprep.plot.scatter(x=adata_CD8_2_magic[:, ['CD4']].to_df(), y=adata_CD8_2_magic[:, ['CD8A']].to_df(), c=adata_CD8_2_magic[:, ['MEOX1']].to_df(),cmap=mymap,  ax=ax2,
                    xlabel='CD4 Normalized Log2(TPM) (Imputed)', ylabel='CD8A Normalized Count  Log2(TPM) (Imputed) ', legend_title="MEOX1", title='CD4/CD8 After MAGIC')

plt.tight_layout()
plt.show()
```


![png](output_110_0.png)


# Metrics


```python
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(33/2.54, 17/2.54))
adata.obs['SampleID'].value_counts().sort_index().plot.bar(x=adata.obs['SampleID'].value_counts().sort_index().index, y=adata.obs['SampleID'].value_counts().sort_index(),color=adata_CD8.uns['SampleID_colors'],ax = ax1)
adata_CD8_2.obs['SampleID'].value_counts().sort_index().plot.bar(x=adata_CD8_2.obs['SampleID'].value_counts().sort_index().index, y=adata_CD8_2.obs['SampleID'].value_counts().sort_index(),color=adata_CD8_2.uns['SampleID_colors'],ax = ax2)
ax1.set_ylabel("SampleID Cell count")
ax1.set_title("TME Cells")
ax2.set_title("CD8+ Cells")


plt.tight_layout()
plt.show()
```


![png](output_112_0.png)



```python
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(33/2.54, 17/2.54))
adata.obs['Therapy'].value_counts().sort_index().plot.bar(x=adata.obs['Therapy'].value_counts().sort_index().index, y=adata.obs['Therapy'].value_counts().sort_index(),ax = ax1)
adata_CD8_2.obs['Therapy'].value_counts().sort_index().plot.bar(x=adata_CD8_2.obs['Therapy'].value_counts().sort_index().index, y=adata_CD8_2.obs['Therapy'].value_counts().sort_index(),ax = ax2)
ax1.set_ylabel("Therapy Cell count")
ax1.set_title("TME Cells")
ax2.set_title("CD8+ Cells")

plt.tight_layout()
plt.show()
```


![png](output_113_0.png)



```python
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(33/2.54, 17/2.54))
adata.obs['Outcome'].value_counts().sort_index().plot.bar(x=adata.obs['Outcome'].value_counts().sort_index().index, y=adata.obs['Outcome'].value_counts().sort_index(),rot=90,ax = ax1)
adata_CD8_2.obs['Outcome'].value_counts().sort_index().plot.bar(x=adata_CD8_2.obs['Outcome'].value_counts().sort_index().index, y=adata_CD8_2.obs['Outcome'].value_counts().sort_index(),rot=90,ax = ax2)
ax1.set_ylabel("Outcome  Cell count")
ax1.set_title("TME Cells")
ax2.set_title("CD8+ Cells")
plt.tight_layout()
plt.show()
```


![png](output_114_0.png)


# Clustering with Louvain algorithm (Similar to Phenograph approach)


```python
sc.tl.louvain(adata_CD8_2_magic,resolution=0.8, key_added='louvain_r0.8',random_state=10,use_weights=True)
sc.tl.louvain(adata_CD8_2_magic, resolution=0.6, key_added='louvain_r0.6', random_state=10,use_weights=True)
sc.tl.louvain(adata_CD8_2_magic, resolution=0.4, key_added='louvain_r0.4', random_state=10,use_weights=True)
sc.tl.louvain(adata_CD8_2_magic, resolution=0.2, key_added='louvain_r0.2', random_state=10,use_weights=True)
```

    running Louvain clustering
        using the "louvain" package of Traag (2017)
        finished: found 10 clusters and added
        'louvain_r0.8', the cluster labels (adata.obs, categorical) (0:00:00)
    running Louvain clustering
        using the "louvain" package of Traag (2017)
        finished: found 6 clusters and added
        'louvain_r0.6', the cluster labels (adata.obs, categorical) (0:00:00)
    running Louvain clustering
        using the "louvain" package of Traag (2017)
        finished: found 6 clusters and added
        'louvain_r0.4', the cluster labels (adata.obs, categorical) (0:00:00)
    running Louvain clustering
        using the "louvain" package of Traag (2017)
        finished: found 3 clusters and added
        'louvain_r0.2', the cluster labels (adata.obs, categorical) (0:00:00)



```python
sc.pl.umap(adata_CD8_2_magic, color=['louvain_r0.8', 'louvain_r0.6','louvain_r0.4','louvain_r0.2'])
```


![png](output_117_0.png)



```python
sc.pl.umap(adata_CD8_2_magic, color=['CD8A','CD4','MEOX1'], use_raw=False, color_map=mymap)
```


![png](output_118_0.png)


# Check Expression of MEOX1 over different Louvain clustering resolution


```python
sc.pl.violin(adata_CD8_2_magic,['MEOX1'], groupby='louvain_r0.8')
sc.pl.violin(adata_CD8_2_magic,['MEOX1'], groupby='louvain_r0.6')
sc.pl.violin(adata_CD8_2_magic,['MEOX1'], groupby='louvain_r0.4')
sc.pl.violin(adata_CD8_2_magic,['MEOX1'], groupby='louvain_r0.2')
```


![png](output_120_0.png)



![png](output_120_1.png)



![png](output_120_2.png)



![png](output_120_3.png)


# Compute DEGs for Louvain Res=0.8 


```python
sc.tl.rank_genes_groups(adata_CD8_2_magic, 'louvain_r0.8', method='wilcoxon')
```

    ranking genes
        finished: added to `.uns['rank_genes_groups']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:46)



```python
pd.DataFrame(adata_CD8_2_magic.uns['rank_genes_groups']['names']).head(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CXCR4</td>
      <td>CCL5</td>
      <td>MALAT1</td>
      <td>DUSP4</td>
      <td>STMN1</td>
      <td>RP5-940J5.9</td>
      <td>SLC2A3</td>
      <td>GZMH</td>
      <td>CCL5</td>
      <td>CCR7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TNFAIP3</td>
      <td>NKG7</td>
      <td>RPL13</td>
      <td>DNAJA1</td>
      <td>HMGN2</td>
      <td>GAPDH</td>
      <td>GPR183</td>
      <td>GNLY</td>
      <td>CCL4</td>
      <td>RPL13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TSC22D3</td>
      <td>PDCD1</td>
      <td>RPS14</td>
      <td>HSP90AB1</td>
      <td>TUBB</td>
      <td>CD27</td>
      <td>FOS</td>
      <td>FGFBP2</td>
      <td>hsa-mir-6723</td>
      <td>TCF7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MALAT1</td>
      <td>PTPRCAP</td>
      <td>IL7R</td>
      <td>HSP90AA1</td>
      <td>TUBA1B</td>
      <td>TIGIT</td>
      <td>EEF1A1</td>
      <td>FCGR3A</td>
      <td>HLA-J</td>
      <td>RPS3A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ZFP36L2</td>
      <td>TRAC</td>
      <td>RPL3</td>
      <td>UBC</td>
      <td>TYMS</td>
      <td>HLA-DRA</td>
      <td>ANXA1</td>
      <td>PXN</td>
      <td>CCL4L2</td>
      <td>EEF1A1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>YPEL5</td>
      <td>PRF1</td>
      <td>RPS3</td>
      <td>VCAM1</td>
      <td>RP11-386G11.10</td>
      <td>RGS1</td>
      <td>TCF7</td>
      <td>NKG7</td>
      <td>DUSP1</td>
      <td>RPS6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NR4A2</td>
      <td>CD8A</td>
      <td>RPLP2</td>
      <td>UBB</td>
      <td>KIAA0101</td>
      <td>FKBP1A</td>
      <td>EEF1A1P5</td>
      <td>TGFBR3</td>
      <td>CCL4L1</td>
      <td>RPL32</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TMEM66</td>
      <td>CD38</td>
      <td>RPL41</td>
      <td>HSPH1</td>
      <td>RPL12P38</td>
      <td>HAVCR2</td>
      <td>BTG2</td>
      <td>PLAC8</td>
      <td>CCL3</td>
      <td>RPL3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CREM</td>
      <td>GBP5</td>
      <td>RPS12</td>
      <td>CREM</td>
      <td>RP5-940J5.9</td>
      <td>CALM3</td>
      <td>GZMK</td>
      <td>S1PR5</td>
      <td>GZMA</td>
      <td>EEF1A1P5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ZFP36</td>
      <td>HAVCR2</td>
      <td>RPL13A</td>
      <td>HSPB1</td>
      <td>GAPDH</td>
      <td>SNAP47</td>
      <td>CD69</td>
      <td>FLNA</td>
      <td>RGS1</td>
      <td>RPS12</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ZNF331</td>
      <td>SIRPG</td>
      <td>RPL10</td>
      <td>HSPD1</td>
      <td>HMGB2</td>
      <td>UCP2</td>
      <td>IL7R</td>
      <td>KLRD1</td>
      <td>CST7</td>
      <td>RPL4</td>
    </tr>
    <tr>
      <th>11</th>
      <td>TMEM2</td>
      <td>STAT1</td>
      <td>TCF7</td>
      <td>HSPA8</td>
      <td>HMGN2P5</td>
      <td>HLA-DQA1</td>
      <td>ZFP36</td>
      <td>CX3CR1</td>
      <td>CD69</td>
      <td>RPS14</td>
    </tr>
    <tr>
      <th>12</th>
      <td>JUNB</td>
      <td>CCL3</td>
      <td>SORL1</td>
      <td>HSPA1A</td>
      <td>H2AFZ</td>
      <td>SERPINB1</td>
      <td>AHNAK</td>
      <td>B2M</td>
      <td>GZMK</td>
      <td>RPS3AP26</td>
    </tr>
    <tr>
      <th>13</th>
      <td>MT-ND2</td>
      <td>HLA-A</td>
      <td>RPS18</td>
      <td>RP11-138I1.4</td>
      <td>MCM7</td>
      <td>HLA-DRB5</td>
      <td>REL</td>
      <td>RPL3</td>
      <td>CCL3L3</td>
      <td>EEF1G</td>
    </tr>
    <tr>
      <th>14</th>
      <td>EZR</td>
      <td>LCP2</td>
      <td>RPL23A</td>
      <td>DNAJB1</td>
      <td>RRM2</td>
      <td>CD38</td>
      <td>CD44</td>
      <td>GPR56</td>
      <td>CD3G</td>
      <td>IL7R</td>
    </tr>
    <tr>
      <th>15</th>
      <td>RNF19A</td>
      <td>CD2</td>
      <td>EEF1A1</td>
      <td>UBBP4</td>
      <td>MKI67</td>
      <td>TNFRSF9</td>
      <td>JUNB</td>
      <td>GZMB</td>
      <td>TRBC2</td>
      <td>RPL10</td>
    </tr>
    <tr>
      <th>16</th>
      <td>SRGN</td>
      <td>RP11-94L15.2</td>
      <td>CTD-2031P19.4</td>
      <td>NOP58</td>
      <td>PPIA</td>
      <td>FCRL3</td>
      <td>CXCR4</td>
      <td>S1PR1</td>
      <td>DNAJB1</td>
      <td>RPS3</td>
    </tr>
    <tr>
      <th>17</th>
      <td>MT-ND1</td>
      <td>PTPN6</td>
      <td>RPL9P8</td>
      <td>HSPE1</td>
      <td>PCNA</td>
      <td>SIRPG</td>
      <td>TMSB4X</td>
      <td>FCGR3B</td>
      <td>IL32</td>
      <td>RPL9P9</td>
    </tr>
    <tr>
      <th>18</th>
      <td>MT-CYB</td>
      <td>CXCR6</td>
      <td>TXNIP</td>
      <td>OAZ1</td>
      <td>ACTB</td>
      <td>GBP2</td>
      <td>VIM</td>
      <td>PRF1</td>
      <td>HLA-H</td>
      <td>RPL9P8</td>
    </tr>
    <tr>
      <th>19</th>
      <td>TUBA4A</td>
      <td>CD27</td>
      <td>RPL9P9</td>
      <td>HAVCR2</td>
      <td>CKS1B</td>
      <td>RPL12</td>
      <td>STAT4</td>
      <td>BIN2</td>
      <td>HLA-L</td>
      <td>RPS3AP6</td>
    </tr>
    <tr>
      <th>20</th>
      <td>PPP2R5C</td>
      <td>IKZF3</td>
      <td>RPS6</td>
      <td>SAMSN1</td>
      <td>HMGB1P5</td>
      <td>ITM2A</td>
      <td>EML4</td>
      <td>CALM1</td>
      <td>KLRK1</td>
      <td>RPL7</td>
    </tr>
    <tr>
      <th>21</th>
      <td>JUND</td>
      <td>TIGIT</td>
      <td>RPS25</td>
      <td>HERPUD1</td>
      <td>NUSAP1</td>
      <td>CD2BP2</td>
      <td>TUBA4A</td>
      <td>SPON2</td>
      <td>MALAT1</td>
      <td>RPL19</td>
    </tr>
    <tr>
      <th>22</th>
      <td>CLK1</td>
      <td>IFNG</td>
      <td>RPL28</td>
      <td>ENTPD1</td>
      <td>DUT</td>
      <td>PARK7</td>
      <td>TMSB4XP8</td>
      <td>FAM65B</td>
      <td>KLF6</td>
      <td>CD55</td>
    </tr>
    <tr>
      <th>23</th>
      <td>TSPYL2</td>
      <td>GZMA</td>
      <td>RPL30</td>
      <td>CHORDC1</td>
      <td>TK1</td>
      <td>TOX</td>
      <td>YPEL5</td>
      <td>IER2</td>
      <td>PRDM1</td>
      <td>RPS4X</td>
    </tr>
    <tr>
      <th>24</th>
      <td>FOSL2</td>
      <td>IL2RB</td>
      <td>RPL10P3</td>
      <td>STAT3</td>
      <td>TMPO</td>
      <td>ATP5G2</td>
      <td>SELK</td>
      <td>LITAF</td>
      <td>TRAC</td>
      <td>RPS13</td>
    </tr>
  </tbody>
</table>
</div>



# Compute DEGs for Louvain Res=0.6


```python
sc.tl.rank_genes_groups(adata_CD8_2_magic, 'louvain_r0.6', method='wilcoxon')
```

    ranking genes
        finished: added to `.uns['rank_genes_groups']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:37)



```python
pd.DataFrame(adata_CD8_2_magic.uns['rank_genes_groups']['names']).head(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CCL5</td>
      <td>MALAT1</td>
      <td>CXCR4</td>
      <td>RP5-940J5.9</td>
      <td>DUSP4</td>
      <td>FLNA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PTPRCAP</td>
      <td>RPL13</td>
      <td>TNFAIP3</td>
      <td>GAPDH</td>
      <td>VCAM1</td>
      <td>GNLY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NKG7</td>
      <td>RPS3</td>
      <td>TSC22D3</td>
      <td>STMN1</td>
      <td>DNAJA1</td>
      <td>TGFBR3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TRAC</td>
      <td>RPL3</td>
      <td>MALAT1</td>
      <td>HMGN2</td>
      <td>HSP90AB1</td>
      <td>FGFBP2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PDCD1</td>
      <td>RPS14</td>
      <td>NR4A2</td>
      <td>TUBB</td>
      <td>HAVCR2</td>
      <td>PXN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>GBP5</td>
      <td>RPL13A</td>
      <td>YPEL5</td>
      <td>TUBA1B</td>
      <td>UBC</td>
      <td>PLAC8</td>
    </tr>
    <tr>
      <th>6</th>
      <td>RAC2</td>
      <td>TCF7</td>
      <td>ZFP36L2</td>
      <td>RP11-386G11.10</td>
      <td>HSPB1</td>
      <td>FOSL2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>PRF1</td>
      <td>RPS12</td>
      <td>TMEM66</td>
      <td>TYMS</td>
      <td>HSP90AA1</td>
      <td>GZMH</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SIRPG</td>
      <td>RPL10</td>
      <td>ZFP36</td>
      <td>RPL12P38</td>
      <td>CREM</td>
      <td>S1PR1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CD8A</td>
      <td>IL7R</td>
      <td>ZNF331</td>
      <td>ACTB</td>
      <td>HSPA1A</td>
      <td>FCGR3A</td>
    </tr>
    <tr>
      <th>10</th>
      <td>CD38</td>
      <td>RPL41</td>
      <td>CREM</td>
      <td>KIAA0101</td>
      <td>RGS2</td>
      <td>B2M</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GZMA</td>
      <td>RPLP2</td>
      <td>JUNB</td>
      <td>PPIA</td>
      <td>HSPA8</td>
      <td>ANXA1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>LCP2</td>
      <td>EEF1A1</td>
      <td>TMEM2</td>
      <td>HMGN2P5</td>
      <td>HSPH1</td>
      <td>S1PR5</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ACTB</td>
      <td>RPS3A</td>
      <td>EIF1</td>
      <td>FABP5</td>
      <td>PHLDA1</td>
      <td>LITAF</td>
    </tr>
    <tr>
      <th>14</th>
      <td>HLA-A</td>
      <td>RPL9P8</td>
      <td>TUBA4A</td>
      <td>HMGB2</td>
      <td>UBB</td>
      <td>MALAT1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>STAT1</td>
      <td>RPL9P9</td>
      <td>TSPYL2</td>
      <td>TPI1</td>
      <td>CTLA4</td>
      <td>IER2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>CD2</td>
      <td>RPL23A</td>
      <td>EZR</td>
      <td>CALM3</td>
      <td>TNFRSF9</td>
      <td>RPL3</td>
    </tr>
    <tr>
      <th>17</th>
      <td>GIMAP4</td>
      <td>RPS25</td>
      <td>RNF19A</td>
      <td>PFN1</td>
      <td>ENTPD1</td>
      <td>EEF1A1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>PTPN6</td>
      <td>RPS6</td>
      <td>SRGN</td>
      <td>H2AFZ</td>
      <td>HSPD1</td>
      <td>CALM1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>RP11-94L15.2</td>
      <td>SORL1</td>
      <td>JUND</td>
      <td>MCM7</td>
      <td>STAT3</td>
      <td>STAT4</td>
    </tr>
    <tr>
      <th>20</th>
      <td>CD27</td>
      <td>CTD-2031P19.4</td>
      <td>PPP2R5C</td>
      <td>HMGB1P5</td>
      <td>DNAJB1</td>
      <td>FAM65B</td>
    </tr>
    <tr>
      <th>21</th>
      <td>CCL3</td>
      <td>RPS18</td>
      <td>FOSL2</td>
      <td>PPIAP22</td>
      <td>CHORDC1</td>
      <td>P2RY8</td>
    </tr>
    <tr>
      <th>22</th>
      <td>IL32</td>
      <td>RPL32</td>
      <td>CLK1</td>
      <td>MKI67</td>
      <td>RP11-138I1.4</td>
      <td>SLC2A3</td>
    </tr>
    <tr>
      <th>23</th>
      <td>RARRES3</td>
      <td>RPL27A</td>
      <td>MT-ND1</td>
      <td>RAN</td>
      <td>NOP58</td>
      <td>BIN2</td>
    </tr>
    <tr>
      <th>24</th>
      <td>IKZF3</td>
      <td>RPL30</td>
      <td>MT-CYB</td>
      <td>PCNA</td>
      <td>NAB1</td>
      <td>CX3CR1</td>
    </tr>
  </tbody>
</table>
</div>



# Compute DEGs for Louvain Res=0.4


```python
sc.tl.rank_genes_groups(adata_CD8_2_magic, 'louvain_r0.4', method='wilcoxon')
```

    ranking genes
        finished: added to `.uns['rank_genes_groups']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:37)



```python
pd.DataFrame(adata_CD8_2_magic.uns['rank_genes_groups']['names']).head(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NKG7</td>
      <td>CXCR4</td>
      <td>MALAT1</td>
      <td>STMN1</td>
      <td>HSP90AB1</td>
      <td>GNLY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CCL5</td>
      <td>TNFAIP3</td>
      <td>RPL13</td>
      <td>HMGN2</td>
      <td>DUSP4</td>
      <td>FLNA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PTPRCAP</td>
      <td>TSC22D3</td>
      <td>RPS3</td>
      <td>TUBA1B</td>
      <td>DNAJA1</td>
      <td>FGFBP2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PRF1</td>
      <td>MALAT1</td>
      <td>RPS14</td>
      <td>TUBB</td>
      <td>HSP90AA1</td>
      <td>PXN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PDCD1</td>
      <td>NR4A2</td>
      <td>RPL3</td>
      <td>RP11-386G11.10</td>
      <td>VCAM1</td>
      <td>FCGR3A</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HAVCR2</td>
      <td>YPEL5</td>
      <td>RPL13A</td>
      <td>TYMS</td>
      <td>HSPB1</td>
      <td>GZMH</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TRAC</td>
      <td>TMEM66</td>
      <td>TCF7</td>
      <td>RP5-940J5.9</td>
      <td>HSPH1</td>
      <td>TGFBR3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CD38</td>
      <td>ZFP36L2</td>
      <td>RPS12</td>
      <td>GAPDH</td>
      <td>HSPA1A</td>
      <td>PLAC8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SIRPG</td>
      <td>ZFP36</td>
      <td>RPL10</td>
      <td>RPL12P38</td>
      <td>HSPA8</td>
      <td>FOSL2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CD8A</td>
      <td>ZNF331</td>
      <td>EEF1A1</td>
      <td>KIAA0101</td>
      <td>CREM</td>
      <td>S1PR1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>GBP5</td>
      <td>CREM</td>
      <td>IL7R</td>
      <td>HMGN2P5</td>
      <td>UBC</td>
      <td>S1PR5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>CCL3</td>
      <td>JUNB</td>
      <td>RPLP2</td>
      <td>PPIA</td>
      <td>UBB</td>
      <td>LITAF</td>
    </tr>
    <tr>
      <th>12</th>
      <td>RAC2</td>
      <td>TMEM2</td>
      <td>RPL41</td>
      <td>HMGB2</td>
      <td>HSPD1</td>
      <td>B2M</td>
    </tr>
    <tr>
      <th>13</th>
      <td>CD27</td>
      <td>TUBA4A</td>
      <td>RPL23A</td>
      <td>H2AFZ</td>
      <td>DNAJB1</td>
      <td>MALAT1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>TIGIT</td>
      <td>EIF1</td>
      <td>RPS3A</td>
      <td>MCM7</td>
      <td>RP11-138I1.4</td>
      <td>ANXA1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>RP11-94L15.2</td>
      <td>RNF19A</td>
      <td>RPS6</td>
      <td>MKI67</td>
      <td>HSPE1</td>
      <td>BIN2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PTPN6</td>
      <td>PPP2R5C</td>
      <td>SORL1</td>
      <td>ACTB</td>
      <td>AHSA1</td>
      <td>CALM1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ACTB</td>
      <td>TSPYL2</td>
      <td>RPL9P8</td>
      <td>RRM2</td>
      <td>CHORDC1</td>
      <td>NKG7</td>
    </tr>
    <tr>
      <th>18</th>
      <td>IKZF3</td>
      <td>EZR</td>
      <td>RPL9P9</td>
      <td>HMGB1P5</td>
      <td>HAVCR2</td>
      <td>KLRD1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>RGS1</td>
      <td>SRGN</td>
      <td>CTD-2031P19.4</td>
      <td>PCNA</td>
      <td>PHLDA1</td>
      <td>P2RY8</td>
    </tr>
    <tr>
      <th>20</th>
      <td>HLA-A</td>
      <td>MT-ND2</td>
      <td>RPS25</td>
      <td>FABP5</td>
      <td>ENTPD1</td>
      <td>RPL3</td>
    </tr>
    <tr>
      <th>21</th>
      <td>IFNG</td>
      <td>JUND</td>
      <td>RPL32</td>
      <td>TMPO</td>
      <td>RGS2</td>
      <td>CX3CR1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>CD2</td>
      <td>MT-ND1</td>
      <td>RPS18</td>
      <td>CKS1B</td>
      <td>CACYBP</td>
      <td>IER2</td>
    </tr>
    <tr>
      <th>23</th>
      <td>CXCR6</td>
      <td>CLK1</td>
      <td>RPL28</td>
      <td>PPIAP22</td>
      <td>SAMSN1</td>
      <td>FAM65B</td>
    </tr>
    <tr>
      <th>24</th>
      <td>GZMA</td>
      <td>NR4A3</td>
      <td>RPL30</td>
      <td>NUSAP1</td>
      <td>UBBP4</td>
      <td>EEF1A1</td>
    </tr>
  </tbody>
</table>
</div>



# Compute DEGs for Louvain Res=0.2


```python
sc.tl.rank_genes_groups(adata_CD8_2_magic, 'louvain_r0.4', method='wilcoxon')
```

    ranking genes
        finished: added to `.uns['rank_genes_groups']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:37)



```python
pd.DataFrame(adata_CD8_2_magic.uns['rank_genes_groups']['names']).head(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NKG7</td>
      <td>CXCR4</td>
      <td>MALAT1</td>
      <td>STMN1</td>
      <td>HSP90AB1</td>
      <td>GNLY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CCL5</td>
      <td>TNFAIP3</td>
      <td>RPL13</td>
      <td>HMGN2</td>
      <td>DUSP4</td>
      <td>FLNA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PTPRCAP</td>
      <td>TSC22D3</td>
      <td>RPS3</td>
      <td>TUBA1B</td>
      <td>DNAJA1</td>
      <td>FGFBP2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PRF1</td>
      <td>MALAT1</td>
      <td>RPS14</td>
      <td>TUBB</td>
      <td>HSP90AA1</td>
      <td>PXN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PDCD1</td>
      <td>NR4A2</td>
      <td>RPL3</td>
      <td>RP11-386G11.10</td>
      <td>VCAM1</td>
      <td>FCGR3A</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HAVCR2</td>
      <td>YPEL5</td>
      <td>RPL13A</td>
      <td>TYMS</td>
      <td>HSPB1</td>
      <td>GZMH</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TRAC</td>
      <td>TMEM66</td>
      <td>TCF7</td>
      <td>RP5-940J5.9</td>
      <td>HSPH1</td>
      <td>TGFBR3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CD38</td>
      <td>ZFP36L2</td>
      <td>RPS12</td>
      <td>GAPDH</td>
      <td>HSPA1A</td>
      <td>PLAC8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SIRPG</td>
      <td>ZFP36</td>
      <td>RPL10</td>
      <td>RPL12P38</td>
      <td>HSPA8</td>
      <td>FOSL2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CD8A</td>
      <td>ZNF331</td>
      <td>EEF1A1</td>
      <td>KIAA0101</td>
      <td>CREM</td>
      <td>S1PR1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>GBP5</td>
      <td>CREM</td>
      <td>IL7R</td>
      <td>HMGN2P5</td>
      <td>UBC</td>
      <td>S1PR5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>CCL3</td>
      <td>JUNB</td>
      <td>RPLP2</td>
      <td>PPIA</td>
      <td>UBB</td>
      <td>LITAF</td>
    </tr>
    <tr>
      <th>12</th>
      <td>RAC2</td>
      <td>TMEM2</td>
      <td>RPL41</td>
      <td>HMGB2</td>
      <td>HSPD1</td>
      <td>B2M</td>
    </tr>
    <tr>
      <th>13</th>
      <td>CD27</td>
      <td>TUBA4A</td>
      <td>RPL23A</td>
      <td>H2AFZ</td>
      <td>DNAJB1</td>
      <td>MALAT1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>TIGIT</td>
      <td>EIF1</td>
      <td>RPS3A</td>
      <td>MCM7</td>
      <td>RP11-138I1.4</td>
      <td>ANXA1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>RP11-94L15.2</td>
      <td>RNF19A</td>
      <td>RPS6</td>
      <td>MKI67</td>
      <td>HSPE1</td>
      <td>BIN2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PTPN6</td>
      <td>PPP2R5C</td>
      <td>SORL1</td>
      <td>ACTB</td>
      <td>AHSA1</td>
      <td>CALM1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ACTB</td>
      <td>TSPYL2</td>
      <td>RPL9P8</td>
      <td>RRM2</td>
      <td>CHORDC1</td>
      <td>NKG7</td>
    </tr>
    <tr>
      <th>18</th>
      <td>IKZF3</td>
      <td>EZR</td>
      <td>RPL9P9</td>
      <td>HMGB1P5</td>
      <td>HAVCR2</td>
      <td>KLRD1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>RGS1</td>
      <td>SRGN</td>
      <td>CTD-2031P19.4</td>
      <td>PCNA</td>
      <td>PHLDA1</td>
      <td>P2RY8</td>
    </tr>
    <tr>
      <th>20</th>
      <td>HLA-A</td>
      <td>MT-ND2</td>
      <td>RPS25</td>
      <td>FABP5</td>
      <td>ENTPD1</td>
      <td>RPL3</td>
    </tr>
    <tr>
      <th>21</th>
      <td>IFNG</td>
      <td>JUND</td>
      <td>RPL32</td>
      <td>TMPO</td>
      <td>RGS2</td>
      <td>CX3CR1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>CD2</td>
      <td>MT-ND1</td>
      <td>RPS18</td>
      <td>CKS1B</td>
      <td>CACYBP</td>
      <td>IER2</td>
    </tr>
    <tr>
      <th>23</th>
      <td>CXCR6</td>
      <td>CLK1</td>
      <td>RPL28</td>
      <td>PPIAP22</td>
      <td>SAMSN1</td>
      <td>FAM65B</td>
    </tr>
    <tr>
      <th>24</th>
      <td>GZMA</td>
      <td>NR4A3</td>
      <td>RPL30</td>
      <td>NUSAP1</td>
      <td>UBBP4</td>
      <td>EEF1A1</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

# Clustering with Leiden algorithm (Similar to Seurat approach)


```python
sc.tl.leiden(adata_CD8_2_magic,resolution=0.8, key_added='leiden_r0.8',random_state=10,use_weights=True)
sc.tl.leiden(adata_CD8_2_magic, resolution=0.6, key_added='leiden_r0.6', random_state=10,use_weights=True)
sc.tl.leiden(adata_CD8_2_magic, resolution=0.4, key_added='leiden_r0.4', random_state=10,use_weights=True)
sc.tl.leiden(adata_CD8_2_magic, resolution=0.2, key_added='leiden_r0.2', random_state=10,use_weights=True)
```

    running Leiden clustering
        finished: found 10 clusters and added
        'leiden_r0.8', the cluster labels (adata.obs, categorical) (0:00:01)
    running Leiden clustering
        finished: found 7 clusters and added
        'leiden_r0.6', the cluster labels (adata.obs, categorical) (0:00:00)
    running Leiden clustering
        finished: found 5 clusters and added
        'leiden_r0.4', the cluster labels (adata.obs, categorical) (0:00:00)
    running Leiden clustering
        finished: found 3 clusters and added
        'leiden_r0.2', the cluster labels (adata.obs, categorical) (0:00:01)



```python
sc.pl.umap(adata_CD8_2_magic, color=['leiden_r0.8', 'leiden_r0.6','leiden_r0.4','leiden_r0.2'])
```


![png](output_136_0.png)



```python
sc.pl.umap(adata_CD8_2_magic, color=['CD8A','CD4','MEOX1'], use_raw=False, color_map=mymap)
```


![png](output_137_0.png)



```python
sc.pl.violin(adata_CD8_2_magic,['MEOX1'], groupby='leiden_r0.8')
sc.pl.violin(adata_CD8_2_magic,['MEOX1'], groupby='leiden_r0.6')
sc.pl.violin(adata_CD8_2_magic,['MEOX1'], groupby='leiden_r0.4')
sc.pl.violin(adata_CD8_2_magic,['MEOX1'], groupby='leiden_r0.2')
```


![png](output_138_0.png)



![png](output_138_1.png)



![png](output_138_2.png)



![png](output_138_3.png)


# Compute DEGs for Leiden Res=0.8


```python
sc.tl.rank_genes_groups(adata_CD8_2_magic, 'leiden_r0.8', method='wilcoxon')
```

    ranking genes
        finished: added to `.uns['rank_genes_groups']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:45)



```python
pd.DataFrame(adata_CD8_2_magic.uns['rank_genes_groups']['names']).head(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CD27</td>
      <td>NR4A2</td>
      <td>MALAT1</td>
      <td>CXCR4</td>
      <td>PDCD1</td>
      <td>CCL5</td>
      <td>DUSP4</td>
      <td>STMN1</td>
      <td>CD69</td>
      <td>GZMH</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RP5-940J5.9</td>
      <td>YPEL5</td>
      <td>RPL13</td>
      <td>TSC22D3</td>
      <td>NKG7</td>
      <td>GZMA</td>
      <td>DNAJA1</td>
      <td>TUBA1B</td>
      <td>MYL12A</td>
      <td>GNLY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CD38</td>
      <td>TNFAIP3</td>
      <td>RPS3</td>
      <td>ZFP36L2</td>
      <td>CCL5</td>
      <td>HLA-J</td>
      <td>VCAM1</td>
      <td>HMGN2</td>
      <td>CD52</td>
      <td>FGFBP2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GAPDH</td>
      <td>ZFP36</td>
      <td>RPL3</td>
      <td>MT-ND1</td>
      <td>CD8A</td>
      <td>CCL4</td>
      <td>CREM</td>
      <td>TUBB</td>
      <td>FOS</td>
      <td>FCGR3A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TIGIT</td>
      <td>SLC2A3</td>
      <td>IL7R</td>
      <td>TMEM66</td>
      <td>TRAC</td>
      <td>PTPRCAP</td>
      <td>HSP90AB1</td>
      <td>RP11-386G11.10</td>
      <td>RAC2</td>
      <td>NKG7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HLA-DRA</td>
      <td>ANXA1</td>
      <td>RPS14</td>
      <td>TMEM2</td>
      <td>HAVCR2</td>
      <td>TMSB4X</td>
      <td>UBC</td>
      <td>TYMS</td>
      <td>TMSB4X</td>
      <td>PXN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>HLA-DRB5</td>
      <td>LMNA</td>
      <td>TCF7</td>
      <td>TNFAIP3</td>
      <td>PRF1</td>
      <td>TRBC2</td>
      <td>HSP90AA1</td>
      <td>KIAA0101</td>
      <td>PIM1</td>
      <td>S1PR5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RGS1</td>
      <td>MALAT1</td>
      <td>RPL13A</td>
      <td>MT-ND2</td>
      <td>CCL3</td>
      <td>RNF213</td>
      <td>HSPH1</td>
      <td>RPL12P38</td>
      <td>VIM</td>
      <td>PLAC8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>UCP2</td>
      <td>FTH1</td>
      <td>RPLP2</td>
      <td>MALAT1</td>
      <td>IFI6</td>
      <td>CCL4L2</td>
      <td>UBB</td>
      <td>RP5-940J5.9</td>
      <td>ANXA2</td>
      <td>TGFBR3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>FKBP1A</td>
      <td>EIF1</td>
      <td>RPL10</td>
      <td>RGS1</td>
      <td>HLA-A</td>
      <td>GZMK</td>
      <td>HSPA8</td>
      <td>HMGB2</td>
      <td>RHOF</td>
      <td>FLNA</td>
    </tr>
    <tr>
      <th>10</th>
      <td>CALM3</td>
      <td>CXCR4</td>
      <td>RPS12</td>
      <td>MT-CYB</td>
      <td>SIRPG</td>
      <td>IL32</td>
      <td>HSPB1</td>
      <td>GAPDH</td>
      <td>RPL28</td>
      <td>CX3CR1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>HAVCR2</td>
      <td>MYADM</td>
      <td>RPL41</td>
      <td>RNF19A</td>
      <td>PTPRCAP</td>
      <td>CD3G</td>
      <td>HAVCR2</td>
      <td>HMGN2P5</td>
      <td>ACTB</td>
      <td>PRF1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ITM2A</td>
      <td>DNAJA1</td>
      <td>EEF1A1</td>
      <td>MT-ND4</td>
      <td>CD38</td>
      <td>CCL4L1</td>
      <td>HSPD1</td>
      <td>H2AFZ</td>
      <td>ANXA2P2</td>
      <td>GPR56</td>
    </tr>
    <tr>
      <th>13</th>
      <td>SERPINB1</td>
      <td>NR4A3</td>
      <td>RPS6</td>
      <td>DUSP4</td>
      <td>STAT1</td>
      <td>HLA-DPB1</td>
      <td>HERPUD1</td>
      <td>MCM7</td>
      <td>ACTG1</td>
      <td>RPL3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SIRPG</td>
      <td>EZR</td>
      <td>RPS3A</td>
      <td>SLA</td>
      <td>HLA-H</td>
      <td>TRAC</td>
      <td>HSPA1A</td>
      <td>RRM2</td>
      <td>TMSB4XP8</td>
      <td>B2M</td>
    </tr>
    <tr>
      <th>15</th>
      <td>HLA-DQA1</td>
      <td>CREM</td>
      <td>SORL1</td>
      <td>TSPYL2</td>
      <td>MIR155HG</td>
      <td>GIMAP7</td>
      <td>SAMSN1</td>
      <td>MKI67</td>
      <td>AHNAK</td>
      <td>FCGR3B</td>
    </tr>
    <tr>
      <th>16</th>
      <td>TOX</td>
      <td>FOSL2</td>
      <td>RPL23A</td>
      <td>NR4A2</td>
      <td>CD27</td>
      <td>B2M</td>
      <td>RP11-138I1.4</td>
      <td>PCNA</td>
      <td>ADAM19</td>
      <td>KLRD1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>MTRNR2L8</td>
      <td>TUBA4A</td>
      <td>CTD-2031P19.4</td>
      <td>MT-CO1</td>
      <td>XAF1</td>
      <td>CST7</td>
      <td>DNAJB1</td>
      <td>ACTB</td>
      <td>FAM65B</td>
      <td>SPON2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>SNAP47</td>
      <td>REL</td>
      <td>RPS25</td>
      <td>ZNF331</td>
      <td>GBP5</td>
      <td>MALAT1</td>
      <td>ENTPD1</td>
      <td>HMGB1P5</td>
      <td>IL7R</td>
      <td>S1PR1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>GBP2</td>
      <td>RP11-138A9.2</td>
      <td>RPS18</td>
      <td>MTND2P28</td>
      <td>TIGIT</td>
      <td>CD2</td>
      <td>UBBP4</td>
      <td>PPIA</td>
      <td>TPT1</td>
      <td>BIN2</td>
    </tr>
    <tr>
      <th>20</th>
      <td>MTRNR2L2</td>
      <td>ZNF331</td>
      <td>RPS4X</td>
      <td>YPEL5</td>
      <td>LYST</td>
      <td>DENND2D</td>
      <td>CTLA4</td>
      <td>CKS1B</td>
      <td>IFITM2</td>
      <td>CALM1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>RP11-94L15.2</td>
      <td>IL7R</td>
      <td>RPL9P8</td>
      <td>PPP2R5C</td>
      <td>CD2</td>
      <td>ITGB2</td>
      <td>STAT3</td>
      <td>NUSAP1</td>
      <td>SELPLG</td>
      <td>LITAF</td>
    </tr>
    <tr>
      <th>22</th>
      <td>NKG7</td>
      <td>CD55</td>
      <td>RPL9P9</td>
      <td>SRGN</td>
      <td>PTPN6</td>
      <td>DUSP1</td>
      <td>ZNF331</td>
      <td>TOP2A</td>
      <td>GLIPR2</td>
      <td>GZMB</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PRF1</td>
      <td>DNAJB1</td>
      <td>RPL10P3</td>
      <td>CREM</td>
      <td>CXCR6</td>
      <td>TBC1D10C</td>
      <td>PHLDA1</td>
      <td>DUT</td>
      <td>SORL1</td>
      <td>KLRG1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>RPL12</td>
      <td>JUNB</td>
      <td>CCR7</td>
      <td>PFKFB3</td>
      <td>HLA-C</td>
      <td>HLA-H</td>
      <td>PRDM1</td>
      <td>TK1</td>
      <td>ANXA1</td>
      <td>C1orf21</td>
    </tr>
  </tbody>
</table>
</div>



# Compute DEGs for Leiden Res=0.6


```python
sc.tl.rank_genes_groups(adata_CD8_2_magic, 'leiden_r0.6', method='wilcoxon')
```

    ranking genes
        finished: added to `.uns['rank_genes_groups']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:39)



```python
pd.DataFrame(adata_CD8_2_magic.uns['rank_genes_groups']['names']).head(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NKG7</td>
      <td>CXCR4</td>
      <td>CCL5</td>
      <td>ANXA1</td>
      <td>MALAT1</td>
      <td>DUSP4</td>
      <td>STMN1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CD38</td>
      <td>MALAT1</td>
      <td>NKG7</td>
      <td>EEF1A1</td>
      <td>RPL13</td>
      <td>DNAJA1</td>
      <td>TUBA1B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PTPRCAP</td>
      <td>TNFAIP3</td>
      <td>CCL4</td>
      <td>SLC2A3</td>
      <td>IL7R</td>
      <td>CREM</td>
      <td>HMGN2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACTB</td>
      <td>TSC22D3</td>
      <td>GZMA</td>
      <td>ZFP36</td>
      <td>RPL3</td>
      <td>HSP90AB1</td>
      <td>TUBB</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PRF1</td>
      <td>NR4A2</td>
      <td>HLA-J</td>
      <td>RPL3</td>
      <td>TCF7</td>
      <td>VCAM1</td>
      <td>RP11-386G11.10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RAC2</td>
      <td>TMEM66</td>
      <td>PTPRCAP</td>
      <td>MALAT1</td>
      <td>RPS3</td>
      <td>UBC</td>
      <td>TYMS</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SIRPG</td>
      <td>ZFP36L2</td>
      <td>HLA-H</td>
      <td>EEF1A1P5</td>
      <td>RPS14</td>
      <td>HSP90AA1</td>
      <td>KIAA0101</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CD27</td>
      <td>YPEL5</td>
      <td>CCL4L2</td>
      <td>EIF1</td>
      <td>RPL41</td>
      <td>HSPH1</td>
      <td>RPL12P38</td>
    </tr>
    <tr>
      <th>8</th>
      <td>PTPN6</td>
      <td>ZFP36</td>
      <td>TRAC</td>
      <td>FOSL2</td>
      <td>RPL10</td>
      <td>HSPA8</td>
      <td>RP5-940J5.9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>RP5-940J5.9</td>
      <td>CREM</td>
      <td>CCL3</td>
      <td>FLNA</td>
      <td>RPLP2</td>
      <td>UBB</td>
      <td>HMGB2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>HLA-DRB5</td>
      <td>MT-ND2</td>
      <td>CCL4L1</td>
      <td>NR4A2</td>
      <td>RPL13A</td>
      <td>SAMSN1</td>
      <td>HMGN2P5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>RP11-94L15.2</td>
      <td>MT-ND1</td>
      <td>TRBC2</td>
      <td>FOS</td>
      <td>RPS12</td>
      <td>HSPB1</td>
      <td>GAPDH</td>
    </tr>
    <tr>
      <th>12</th>
      <td>TIGIT</td>
      <td>ZNF331</td>
      <td>CD2</td>
      <td>IER2</td>
      <td>EEF1A1</td>
      <td>HERPUD1</td>
      <td>H2AFZ</td>
    </tr>
    <tr>
      <th>13</th>
      <td>IKZF3</td>
      <td>MT-CYB</td>
      <td>HLA-G</td>
      <td>BTG2</td>
      <td>SORL1</td>
      <td>HSPD1</td>
      <td>RRM2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>GBP5</td>
      <td>JUNB</td>
      <td>CD8A</td>
      <td>AHNAK</td>
      <td>RPS6</td>
      <td>HSPA1A</td>
      <td>MKI67</td>
    </tr>
    <tr>
      <th>15</th>
      <td>PDCD1</td>
      <td>TMEM2</td>
      <td>PDCD1</td>
      <td>CD55</td>
      <td>RPL9P8</td>
      <td>CTLA4</td>
      <td>MCM7</td>
    </tr>
    <tr>
      <th>16</th>
      <td>GAPDH</td>
      <td>TSPYL2</td>
      <td>CD3G</td>
      <td>S1PR1</td>
      <td>RPL9P9</td>
      <td>HAVCR2</td>
      <td>PCNA</td>
    </tr>
    <tr>
      <th>17</th>
      <td>GBP2</td>
      <td>PPP2R5C</td>
      <td>B2M</td>
      <td>TGFBR3</td>
      <td>RPS18</td>
      <td>RP11-138I1.4</td>
      <td>HMGB1P5</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HAVCR2</td>
      <td>EIF1</td>
      <td>RNF213</td>
      <td>JUNB</td>
      <td>RPS3A</td>
      <td>PHLDA1</td>
      <td>PPIA</td>
    </tr>
    <tr>
      <th>19</th>
      <td>TRAC</td>
      <td>JUND</td>
      <td>HLA-DPB1</td>
      <td>NFKBIA</td>
      <td>RPL23A</td>
      <td>NAB1</td>
      <td>ACTB</td>
    </tr>
    <tr>
      <th>20</th>
      <td>PSMB10</td>
      <td>EZR</td>
      <td>HLA-C</td>
      <td>STAT4</td>
      <td>PLAC8</td>
      <td>ENTPD1</td>
      <td>CKS1B</td>
    </tr>
    <tr>
      <th>21</th>
      <td>STAT1</td>
      <td>NR4A3</td>
      <td>APOBEC3C</td>
      <td>PXN</td>
      <td>CTD-2031P19.4</td>
      <td>DNAJB1</td>
      <td>NUSAP1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>FKBP1A</td>
      <td>MTND2P28</td>
      <td>HLA-A</td>
      <td>FTH1</td>
      <td>RPL32</td>
      <td>UBBP4</td>
      <td>DUT</td>
    </tr>
    <tr>
      <th>23</th>
      <td>ACTG1</td>
      <td>RNF19A</td>
      <td>CST7</td>
      <td>NR4A3</td>
      <td>RPS25</td>
      <td>RGS2</td>
      <td>TOP2A</td>
    </tr>
    <tr>
      <th>24</th>
      <td>CCL5</td>
      <td>TUBA4A</td>
      <td>CD74</td>
      <td>CTC-250I14.6</td>
      <td>RPS4X</td>
      <td>AHSA1</td>
      <td>TK1</td>
    </tr>
  </tbody>
</table>
</div>



# Compute DEGs for Leiden Res=0.4


```python
sc.tl.rank_genes_groups(adata_CD8_2_magic, 'leiden_r0.4', method='wilcoxon')
```

    ranking genes
        finished: added to `.uns['rank_genes_groups']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:35)



```python
pd.DataFrame(adata_CD8_2_magic.uns['rank_genes_groups']['names']).head(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CCL5</td>
      <td>MALAT1</td>
      <td>CXCR4</td>
      <td>STMN1</td>
      <td>HSP90AA1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NKG7</td>
      <td>RPL3</td>
      <td>TSC22D3</td>
      <td>HMGN2</td>
      <td>HSP90AB1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HAVCR2</td>
      <td>EEF1A1</td>
      <td>ZFP36L2</td>
      <td>TUBA1B</td>
      <td>HSPH1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PTPRCAP</td>
      <td>RPS12</td>
      <td>TNFAIP3</td>
      <td>TUBB</td>
      <td>UBB</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PDCD1</td>
      <td>RPL13</td>
      <td>TMEM2</td>
      <td>TYMS</td>
      <td>DNAJA1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PRF1</td>
      <td>RPS3</td>
      <td>MT-ND1</td>
      <td>RP11-386G11.10</td>
      <td>HSPD1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SIRPG</td>
      <td>RPL10</td>
      <td>MT-ND2</td>
      <td>KIAA0101</td>
      <td>CREM</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RGS1</td>
      <td>RPS14</td>
      <td>TMEM66</td>
      <td>RPL12P38</td>
      <td>HSPA8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TRAC</td>
      <td>IL7R</td>
      <td>MALAT1</td>
      <td>RP5-940J5.9</td>
      <td>HSPA1A</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CD38</td>
      <td>RPL13A</td>
      <td>RGS1</td>
      <td>GAPDH</td>
      <td>RP11-138I1.4</td>
    </tr>
    <tr>
      <th>10</th>
      <td>CD8A</td>
      <td>TCF7</td>
      <td>MT-CYB</td>
      <td>HMGN2P5</td>
      <td>UBBP4</td>
    </tr>
    <tr>
      <th>11</th>
      <td>TIGIT</td>
      <td>RPS6</td>
      <td>RNF19A</td>
      <td>HMGB2</td>
      <td>DNAJB6</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CD27</td>
      <td>RPLP2</td>
      <td>DUSP4</td>
      <td>MCM7</td>
      <td>HSPE1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>GBP5</td>
      <td>RPL3P4</td>
      <td>MT-ND4</td>
      <td>H2AFZ</td>
      <td>RP11-1033A18.1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>LYST</td>
      <td>EEF1A1P5</td>
      <td>MTND2P28</td>
      <td>PPIA</td>
      <td>HSPB1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>RAC2</td>
      <td>FOS</td>
      <td>ZNF331</td>
      <td>MKI67</td>
      <td>DUSP4</td>
    </tr>
    <tr>
      <th>16</th>
      <td>CXCR6</td>
      <td>RPL23A</td>
      <td>SLA</td>
      <td>PCNA</td>
      <td>DNAJB1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>CCL3</td>
      <td>SORL1</td>
      <td>NR4A2</td>
      <td>ACTB</td>
      <td>CACYBP</td>
    </tr>
    <tr>
      <th>18</th>
      <td>IL32</td>
      <td>RPS18</td>
      <td>CREM</td>
      <td>RRM2</td>
      <td>AHSA1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>SNAP47</td>
      <td>RPS3A</td>
      <td>TSPYL2</td>
      <td>HMGB1P5</td>
      <td>UBC</td>
    </tr>
    <tr>
      <th>20</th>
      <td>GZMA</td>
      <td>S1PR1</td>
      <td>MT-CO1</td>
      <td>CKS1B</td>
      <td>CHORDC1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>RP5-940J5.9</td>
      <td>RPL41</td>
      <td>MT-ND5</td>
      <td>DUT</td>
      <td>YPEL5</td>
    </tr>
    <tr>
      <th>22</th>
      <td>LSP1</td>
      <td>RPS4X</td>
      <td>SRGN</td>
      <td>NUSAP1</td>
      <td>HSPD1P1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>HLA-A</td>
      <td>RPL13AP5</td>
      <td>KLRK1</td>
      <td>PFN1</td>
      <td>SAMSN1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>RP11-94L15.2</td>
      <td>PLAC8</td>
      <td>PFKFB3</td>
      <td>TK1</td>
      <td>NOP58</td>
    </tr>
  </tbody>
</table>
</div>



# Compute DEGs for Leiden Res=0.2


```python
sc.tl.rank_genes_groups(adata_CD8_2_magic, 'leiden_r0.2', method='wilcoxon')
```

    ranking genes
        finished: added to `.uns['rank_genes_groups']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:31)



```python
pd.DataFrame(adata_CD8_2_magic.uns['rank_genes_groups']['names']).head(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CCL5</td>
      <td>MALAT1</td>
      <td>STMN1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NKG7</td>
      <td>RPL3</td>
      <td>HMGN2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HAVCR2</td>
      <td>IL7R</td>
      <td>TUBB</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PRF1</td>
      <td>EEF1A1</td>
      <td>TUBA1B</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PDCD1</td>
      <td>RPL13</td>
      <td>TYMS</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CD27</td>
      <td>RPS3</td>
      <td>RP11-386G11.10</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LYST</td>
      <td>CXCR4</td>
      <td>KIAA0101</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TIGIT</td>
      <td>RPL13A</td>
      <td>RPL12P38</td>
    </tr>
    <tr>
      <th>8</th>
      <td>RGS1</td>
      <td>RPS14</td>
      <td>RP5-940J5.9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CD38</td>
      <td>MT-ND2</td>
      <td>GAPDH</td>
    </tr>
    <tr>
      <th>10</th>
      <td>CD74</td>
      <td>RPS12</td>
      <td>HMGB2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>LSP1</td>
      <td>MT-ND1</td>
      <td>HMGN2P5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>PTPRCAP</td>
      <td>MT-CYB</td>
      <td>MCM7</td>
    </tr>
    <tr>
      <th>13</th>
      <td>SIRPG</td>
      <td>EIF1</td>
      <td>H2AFZ</td>
    </tr>
    <tr>
      <th>14</th>
      <td>GBP5</td>
      <td>RPS3A</td>
      <td>RRM2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>CXCR6</td>
      <td>RPL10</td>
      <td>PCNA</td>
    </tr>
    <tr>
      <th>16</th>
      <td>RP5-940J5.9</td>
      <td>TCF7</td>
      <td>MKI67</td>
    </tr>
    <tr>
      <th>17</th>
      <td>APOBEC3C</td>
      <td>ZFP36L2</td>
      <td>PPIA</td>
    </tr>
    <tr>
      <th>18</th>
      <td>CD8A</td>
      <td>RPLP2</td>
      <td>ACTB</td>
    </tr>
    <tr>
      <th>19</th>
      <td>TRAC</td>
      <td>RPS6</td>
      <td>HMGB1P5</td>
    </tr>
    <tr>
      <th>20</th>
      <td>GAPDH</td>
      <td>RPS27</td>
      <td>CKS1B</td>
    </tr>
    <tr>
      <th>21</th>
      <td>HLA-A</td>
      <td>PABPC1</td>
      <td>DUT</td>
    </tr>
    <tr>
      <th>22</th>
      <td>HLA-DRB5</td>
      <td>MTND2P28</td>
      <td>NUSAP1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>HLA-DRB1</td>
      <td>TMEM66</td>
      <td>PFN1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>CCL3</td>
      <td>JUNB</td>
      <td>TK1</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
