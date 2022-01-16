


function [DATA_OUT] =PCA_DWW(SIGN, DATA_IN, NUM_DIM)

%% 实现PCA降维处理
%   输入：
%         SIGN:PCA实现方法标志变量。
%         DATA_IN：待降维数据矩阵（M*N，M为数据样本数，N为数据维度）
%         NUM_DIM：目标维度
%   输出：
%         DATA_OUT：降维后数据矩阵


switch SIGN
    case 1
        %% 方法一：使用Matlab工具箱princomp函数实现PCA

         [COEFF SCORE latent]=princomp(DATA_IN);
         DATA_OUT =SCORE(:,1:NUM_DIM);   %取前k个主成分
    case 2
        %% 方法二：自编程序实现PCA
        [Row Col]=size(DATA_IN);
        covX=cov(DATA_IN);                                    %求样本的协方差矩阵（散步矩阵除以(n-1)即为协方差矩阵）
        [V D]=eigs(covX);                               %求协方差矩阵的特征值D和特征向量V
        meanX=mean(DATA_IN);                                  %样本均值m
        %所有样本X减去样本均值m，再乘以协方差矩阵（散步矩阵）的特征向量V，即为样本的主成份SCORE
        tempX= repmat(meanX,Row,1);
        SCORE2=(DATA_IN-tempX)*V ;                             %主成份：SCORE
        DATA_OUT=SCORE2(:,1:NUM_DIM);
    case 3
        
        %% 方法三：使用快速PCA算法实现PCA
         [DATA_OUT COEFF3] = fastPCA(DATA_IN, NUM_DIM );
end
        
        
        

















