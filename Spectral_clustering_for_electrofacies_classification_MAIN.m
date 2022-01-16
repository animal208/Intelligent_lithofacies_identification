%%  Spectral_clustering_for_electrofacies_classification
%%  Author:weiwu dong             date:2021/10/25

clear
% read input data
[data] =importdata('input.txt');
[n_sample,n_dimension]=size(data);

data_scale=scale_dww(data,1,0,1,[],[]);

% dimension reduction

no_dims=2;
prp=30;
max_iter=1000;
mom_switch_iter=250;
stop_lying_iter=100;


chose=2;
switch chose
    case 1  % PCA
        data_dr =PCA_DWW(2, data_scale, no_dims);
    case 2  % t-SNE
        [data_dr,cost] = tsne_dww(data_scale,[], no_dims,[], prp,0.5,0.8,mom_switch_iter,stop_lying_iter,max_iter,500,0.01);
end

% Unsupervised spectral clustering analysis
num_clusters=23;
sigma=2;
num_neighbors=20;
block_size=1000;

A=gen_nn_distance_dww(data_dr, num_neighbors, block_size);
        
[cluster_labels evd_time kmeans_time total_time] = sc(A, sigma, num_clusters);
        
% save the results         
dlmwrite('Clustering_results.txt',cluster_labels,'delimiter',' ');

n_cluster_dww=length(unique(cluster_labels));

h=figure('Name','SC Index','NumberTitle','off');
chgicon(h,'2.jpg');  % 更改图标 
[ss,ff]=silhouette(data_dr,cluster_labels); % 绘制聚类轮廓图
title('SC Index，[-1,1]，The bigger the better');
xlabel('SC Index');
ylabel('CLUSTERS');
