### (a) What are the problems with this clustering algorithm?
Since the cluster centers are selected/initialised randomly, there is only little chance to reproduce results. Also the algorithm only finds a local mininum. Since we are using the mean to calculate the (new) centroids we are emphasising outliers, which may lead to crooked results. Furthermore, we have to decide on a reasonable numbers of clusters K before.

### (b) How to improve?
Use a fixed number of clusters (calculate in before); use K-Medoids instead (more robust towards outliers)