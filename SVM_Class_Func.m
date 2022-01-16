%% SVM分类器（基于libsvm-3.1-[FarutoUltimate3.1Mcode]），用于测井岩性划分

% 使用时，将libsvm-3.1-[FarutoUltimate3.1Mcode]文件夹加到path路径下
% 作者：董维武                               2018.8.15

% 输入数据train.txt（可由BOUND.DAT导出）格式：
% 起始深度  终止深度  样本类别  特征属性1  特征属性2  .....

% 输入数据predict.txt（可由ZONE.DAT导出）格式：
% 起始深度  终止深度  1   特征属性1  特征属性2  .....
% predict.txt中包含train.txt（即预测数据中包含了训练样本

% 输入：1、sign_scale：数据归一化方法选择标志变量（scale）；
%                     sign_scale=1，自动选取各列最大、最小值；
%                     sign_scale=2，人为指定各列最大、最小值；
%                     sign_scale=3，不做归一化处理；
%                     
%       2、mindww,maxdww：人为指定的待归一化数据各列最大最小值（行向量）；                    
%       3、sign_drm：数据降维处理方法选择标志变量(dimension reduction method)；
%                     sign_drm=1，PCA降维处理；
%                     sign_drm=2，FASTICA降维处理；
%                     sign_drm=3，不做降维处理；
%       4、sign_pom：c、g参数优选方法选择标志变量(parameter optimization method)；
%                     sign_pom=1，网格参数优化（c，g）；
%                     sign_pom=2，GA参数优化（c，g）；
%                     sign_pom=3，PSO算法参数优化（c，g）；
%                     sign_pom=4，人工选取参数（c，g）；
%       5、sign_data: 建模数据选择标志变量；
%                     sign_data=1，将全部样本数据分为训练数据和测试数据，分别用于建模和测试；
%                     sign_data=2，将全部样本数据都用于建模及测试。

%     例： [model]=svm_dww(1,0,1,[],[],3,1,1)  
%          [model]=svm_dww(2,0,1,[1 55 240 70],[20 230 600 430],3,1,1)

function [model,TYPE2]=SVM_Class_Func(labels,data,stdep,endep,data_predict,...
    sign_scale,sign_scale_save,min_scale,max_scale,mindww,maxdww,sign_drm,sign_drm_save,sign_pom,sign_pso,...
    sign_data,sign_data_save,k_fold,sign_rescale,net_option,ga_option,pso_option,cost,gamma,TYPE_name,svm_option)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                 1、导入数据
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %clear;
% fulldata_train=textread('train.txt');
% labels=fulldata_train(:,3);
% data=fulldata_train(:,4:end);%样本集
% 
% fulldata_predict=textread('predict.txt');
% stdep=fulldata_predict(:,1);
% endep=fulldata_predict(:,2);
% data_predict=fulldata_predict(:,4:end);%待测数据集
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2、归一化预处理，样本集和待测数据集要进行相同的归一化处理
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  方法一：自动选取各列最大、最小值；
if sign_scale==1
    
    [data_scale,data_predict_scale] = scaleForSVM(data,data_predict,min_scale,max_scale);
    % 函数接口：[train_scale,test_scale,ps]=scaleForSVM(train_data,test_data,ymin,ymax)
    


%  方法二：人为指定各列最大、最小值,每列的最大、最小值由maxdww和mindww数组依次给出；    
elseif sign_scale==2
    
    
    [data_scale]=scale_dww(data,2,min_scale,max_scale,mindww,maxdww);
    [data_predict_scale]=scale_dww(data_predict,2,min_scale,max_scale,mindww,maxdww);
    % 函数接口：[data_scale]=scale_dww(data,sign,ymin,ymax,mindww,maxdww)
    
%  不做归一化处理
elseif sign_scale==3
    data_scale=data;
    data_predict_scale=data_predict;
else
    msgbox('sign_scale error!!!!');

end   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%保存规范化后数据
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if sign_scale_save==1
    [mmm1 nnn1]=size(data_scale);
    [mmm2 nnn2]=size(data_predict_scale);
    fp1=fopen('data_train_scale.txt','w');
    fp2=fopen('data_predict_scale.txt','w');
    for i=1:1:mmm1
        fprintf(fp1,'%2d  ',labels(i));
        for j=1:1:nnn1
            fprintf(fp1,'%f  ',data_scale(i,j));
        end
        fprintf(fp1,'\n');
    end

    for i=1:1:mmm2
        fprintf(fp2,'%f  %f  ',stdep(i),endep(i));
        for j=1:1:nnn2
            fprintf(fp2,'%f  ',data_predict_scale(i,j));
        end
        fprintf(fp2,'\n');
    end

    fclose(fp1);
    fclose(fp2);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                3、降维预处理
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 方法一：PCA
if sign_drm==1
    [data_drm,data_predict_drm] = pcaForSVM(data_scale,data_predict_scale,90);
    
    %对降维后的数据重新进行归一化处理
    if sign_rescale==1
        [data_drm,data_predict_drm] = scaleForSVM(data_drm,data_predict_drm,min_scale,max_scale);
    end

    % 函数接口：[train_pca,test_pca] = pcaForSVM(train,test,threshold)
    % 输入： 
    %     train_data：训练集，格式要求与svmtrain相同。 
    %     test_data：测试集，格式要求与svmtrain相同。 
    %     threshold：对原始变量的解释程度（[0，100]之间的一个数） ，通过该阈值可
    %     以选取出主成分，该参数可以不输入，默认为90，即选取的主成分默认可以
    %     达到对原始变量达到 90%的解释程度。 
    % 输出： 
    %     train_pca：进行 pca 降维预处理后的训练集。 
    %     test_pca：进行 pca 降维预处理后的测试集。


% 方法二：FASTICA
elseif sign_drm==2
    [data_drm,data_predict_drm] = fasticaForSVM(data_scale,data_predict_scale);
    
    %对降维后的数据重新进行归一化处理
    if sign_rescale==1
           [data_drm,data_predict_drm] = scaleForSVM(data_drm,data_predict_drm,min_scale,max_scale);
    end
    
%  不做降维处理
elseif sign_drm==3
    data_drm=data_scale;
    data_predict_drm=data_predict_scale;

else
    msgbox('sign_drm error！！！！！');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%保存降维处理后数据
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if sign_drm_save==1
    [mmm1 nnn1]=size(data_drm);
    [mmm2 nnn2]=size(data_predict_drm);
    fp1=fopen('data_train_drm.txt','w');
    fp2=fopen('data_predict_drm.txt','w');
    for i=1:1:mmm1
        fprintf(fp1,'%2d  ',labels(i));
        for j=1:1:nnn1
            fprintf(fp1,'%f  ',data_drm(i,j));
        end
        fprintf(fp1,'\n');
    end


    for i=1:1:mmm2
        fprintf(fp2,'%f  %f  ',stdep(i),endep(i));
        for j=1:1:nnn2
            fprintf(fp2,'%f  ',data_predict_drm(i,j));
        end
        fprintf(fp2,'\n');
    end

    fclose(fp1);
    fclose(fp2);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4、将样本数据随机分组，一组用于测试，其余各组用于训练
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if sign_data==1      %将全部样本数据分为训练数据和测试数据，分别用于建模和测试；
    N=length(labels);
    indices=crossvalind('Kfold',N,k_fold);
    j=1;
    k=1;
    for i=1:1:N
        if indices(i)==1
            test_labels(j)=labels(i);
            test_data(j,:)=data_drm(i,:);
            j=j+1;
        else
            train_labels(k)=labels(i);
            train_data(k,:)=data_drm(i,:);
            k=k+1;
        end
    end
    test_labels=test_labels';
    train_labels=train_labels';
%     m=length(train_labels);
%     n=length(test_labels);
%     dlmwrite('temp1.txt',train_labels);%存储训练样本数据，用于遗传算法中计算适应度函数
%     dlmwrite('temp2.txt',train_data);
    
elseif sign_data==2       %将全部样本数据都用于建模及测试。
    train_labels=labels;
    test_labels=labels;
    train_data=data_drm;
    test_data=data_drm;
    
    
%     dlmwrite('temp1.txt',labels);%存储训练样本数据，用于遗传算法中计算适应度函数
%     dlmwrite('temp2.txt',data);
    
else
    msgbox('sign_data error！！！！！');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%保存训练集，供PSOt_DWW算法调用
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save PSOt_DWW_data.mat train_labels train_data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%保存训练集和测试集
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if sign_data_save==1
    [mmm1 nnn1]=size(train_data);
    [mmm2 nnn2]=size(test_data);
    fp1=fopen('train_data.txt','w');
    fp2=fopen('test_data.txt','w');
    for i=1:1:mmm1
        fprintf(fp1,'%2d  ',train_labels(i));
        for j=1:1:nnn1
            fprintf(fp1,'%f  ',train_data(i,j));
        end
        fprintf(fp1,'\n');
    end


    for i=1:1:mmm2
        fprintf(fp2,'%2  ',test_labels(i));
        for j=1:1:nnn2
            fprintf(fp2,'%f  ',test_data(i,j));
        end
        fprintf(fp2,'\n');
    end

    fclose(fp1);
    fclose(fp2);
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                5、参数c、g优选
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%  方法一：网格搜索参数优化（c，g）

if sign_pom==1
    
    cmin=net_option.cmin;
    cmax=net_option.cmax;
    gmin=net_option.gmin;
    gmax=net_option.gmax;
    v=net_option.v;
    cstep=net_option.cstep;
    gstep=net_option.gstep;
    accstep=net_option.accstep;
    
    
    
    [bestacc,bestc,bestg]=SVMcgForClass(train_labels,train_data,cmin,cmax,...
    gmin,gmax,v,cstep,gstep,accstep,svm_option);
    
%     [bestacc,bestc,bestg]=SVMcgForClass(train_labels,train_data,-10,10,...
%     -10,10,5,1,1,4.5);


%函数接口：[bestacc,bestc,bestg]=SVMcgForClass(train_label,train,cmin,cmax,...
%           gmin,gmax,v,cstep,gstep,accstep)
%
%     输入： 
%         train_label：训练集的标签，格式要求与 svmtrain相同。 
%         train：训练集，格式要求与svmtrain相同。 
%         cmin，cmax：惩罚参数 c 的变化范围，即在[2^cmin，2^cmax]范围内寻找最
%         佳的参数 c，默认值为 cmin=-8，cmax=8，即默认惩罚参数 c 的范围是[2^(-8)，
%         2^8]。 
%         gmin，gmax：RBF核参数 g 的变化范围，即在[2^gmin，2^gmax]范围内寻找
%         最佳的 RBF 核参数 g，默认值为 gmin=-8，gmax=8，即默认 RBF 核参数 g 的范
%         围是[2^(-8)，2^8]。 
%         v：进行 Cross Validation 过程中的参数，即对训练集进行 v-fold Cross 
%         Validation，默认为3，即默认进行 3 折 CV 过程。 
%         cstep，gstep：进行参数寻优是 c 和 g 的步进大小，即 c 的取值为 2^cmin，
%         2^(cmin+cstep)， …， 2^cmax， ， g 的取值为 2^gmin， 2^(gmin+gstep)， …， 2^gmax，
%         默认取值为 cstep=1，gstep=1。 
%         accstep： 最后参数选择结果图中准确率离散化显示的步进间隔大小 （[0， 100]
%         之间的一个数），默认为4.5。 
%      输出： 
%         bestCVaccuracy：最终CV意义下的最佳分类准确率。 
%         bestc：最佳的参数c。 
%         bestg：最佳的参数g。


%%  方法二：GA算法参数优化（c，g）

elseif sign_pom==2
    

%     ga_option.maxgen = 100;
%     ga_option.sizepop = 20; 
%     ga_option.pCrossover = 0.4;
%     ga_option.pMutation = 0.01;
%     ga_option.cbound = [0.1,100];
%     ga_option.gbound = [0.1,100];
%     ga_option.v = 10;  
%     ga_option.ggap = 0.9;
 
    [bestacc,bestc,bestg,ga_option]= gaSVMcgForClass(train_labels,train_data,ga_option,svm_option);
 

%函数接口：[bestCVaccuracy,bestc,bestg,ga_option]= gaSVMcgForClass(train_label,
%           train,ga_option) 
%
%         输入： 
%             train_label：训练集的标签，格式要求与svmtrain相同。 
%             train：训练集，格式要求与svmtrain相同。 
%             ga_option：GA中的一些参数设置，可不输入，有默认值，详细请看代码的帮助说明。
%          
%         输出： 
%             bestCVaccuracy：最终 CV意义下的最佳分类准确率。 
%             bestc：最佳的参数c。 
%             bestg：最佳的参数g。 
%             ga_option：记录 GA中的一些参数。
%
%         ga_option参数结构体：
%             maxgen:最大的进化代数,默认为100,一般取值范围为[100,500]
%             sizepop:种群最大数量,默认为20,一般取值范围为[20,100]
%             pCrossover:交叉概率,默认为0.4,一般取值范围为[0.4,0.99]
%             pMutation:变异概率,默认为0.01,一般取值范围为[0.001,0.1]
%             cbound = [cmin,cmax],参数c的变化范围,默认为[0.1,100]
%             gbound = [gmin,gmax],参数g的变化范围,默认为[0.01,1000]
%             v:SVM Cross Validation参数,默认为3

 

%%  方法三：PSO算法参数优化（c，g）

elseif sign_pom==3

%     pso_option.c1 = 1.5;
%     pso_option.c2 = 1.7;
%     pso_option.maxgen = 100;
%     pso_option.sizepop = 20;
%     pso_option.k = 0.6;
%     pso_option.wV = 1;
%     pso_option.wP = 1;
%     pso_option.v = 3;
%     pso_option.popcmax = 100;
%     pso_option.popcmin = 0.1;
%     pso_option.popgmax = 100;
%     pso_option.popgmin = 0.1;
    
    if sign_pso==0
        [bestacc,bestc,bestg] = psoSVMcgForClass(train_labels,train_data,pso_option,svm_option);
    else
        h=figure;
        chgicon(h,'2.jpg');  % 更改图标
        set(h,'name','粒子群动态分布','Numbertitle','off');        
        c_range=[pso_option.popcmin,pso_option.popcmax];
        g_range=[pso_option.popgmin,pso_option.popgmax];
        range = [c_range;g_range];
        Max_V = 0.2*(range(:,2)-range(:,1));  %最大速度取范围的10%~20%
        n=2;
        psoparams=[1 pso_option.maxgen pso_option.sizepop 2 2 0.9 0.4 1500 1e-8 250 NaN 0 0];
        
        out=pso_Trelea_vectorized('PSOt_DWW_func',n,Max_V,range,1,psoparams);
        bestc=out(1);
        bestg=out(2);
    end
        



%   函数接口：[bestCVaccuracy,bestc,bestg,pso_option]= psoSVMcgForClass(train_label,...
%             train,pso_option) 
%     输入： 
%          train_label：训练集的标签，格式要求与svmtrain相同。 
%          train：训练集，格式要求与svmtrain相同。 
%          pso_option：PSO 中的一些参数设置，可不输入，有默认值，详细请看代码的帮助说明。
%      
%     输出： 
%          bestCVaccuracy：最终 CV意义下的最佳分类准确率。 
%          bestc：最佳的参数c。 
%          bestg：最佳的参数g。
%          pso_option：记录 PSO中的一些参数。
%
%     pso_option参数结构体：
%          c1:初始为1.5,pso参数局部搜索能力
%          c2:初始为1.7,pso参数全局搜索能力
%          maxgen:初始为200,最大进化数量
%          sizepop:初始为20,种群最大数量
%          k:初始为0.6(k belongs to [0.1,1.0]),速率和x的关系(V = kX)
%          wV:初始为1(wV best belongs to [0.8,1.2]),速率更新公式中速度前面的弹性系数
%          wP:初始为1,种群更新公式中速度前面的弹性系数
%          v:初始为3,SVM Cross Validation参数
%          popcmax:初始为100,SVM 参数c的变化的最大值.
%          popcmin:初始为0.1,SVM 参数c的变化的最小值.
%          popgmax:初始为1000,SVM 参数g的变化的最大值.
%          popgmin:初始为0.01,SVM 参数c的变化的最小值.



%% 方法四：人工选取参数（c，g）

elseif sign_pom==4
    bestc=cost;
    bestg=gamma;
 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             6、建模、测试及分类预测
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



svm_type=num2str(svm_option.svm_type);
kernel_type=num2str(svm_option.kernel_type);
degree=num2str(svm_option.degree);
coef0=num2str(svm_option.coef0);
nu=num2str(svm_option.nu);
epsilon=num2str(svm_option.epsilon);
cachesize=num2str(svm_option.cachesize);
eps=num2str(svm_option.eps);
weight=num2str(svm_option.weight);
shrinking=num2str(svm_option.shrinking);
probability_estimates=num2str(svm_option.probability_estimates);

%  建模

cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -s ',svm_type,' -t ',kernel_type,' -d ',degree,' -r ',coef0,' -n ',nu,' -p ',epsilon,' -m ',cachesize,' -e ',eps,' -wi ',weight,' -h ',shrinking,' -b ',probability_estimates];
%    cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -s ',svm_type,' -t ',kernel_type];
% cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];


model = svmtrain(train_labels, train_data,cmd);
% assignin('base','model',model)
%  回判
fprintf('回判:');
[train_predict_labels, train_accuracy] = svmpredict(train_labels, train_data, model);
% train_accuracy(1)

%  测试
fprintf('测试:');
[test_predict_labels, test_accuracy] = svmpredict(test_labels, test_data, model);
% test_accuracy(1)

%  预测
fprintf('预测:\n');
for i=1:1:20
    unknown_labels=ones(length(stdep),1)*i;
    fprintf('%2d:',i);[predict_labels] = svmpredict(unknown_labels, data_predict_drm, model);
end


%%%%%%%%%%%%%%%%%%%%%%%%%
%保存各参数值及运行结果
%%%%%%%%%%%%%%%%%%%%%%%%%



for i=1:1:20
    count(i)=0;
end

mm=length(stdep);
for i=1:1:mm
    for j=1:1:20
        if predict_labels(i)==j
            count(j)=count(j)+1;
        end 
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         7、数据导出，将分类结果输出到文件
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fp=fopen('SVM_identification_results.txt','w');
rlev=0.125;
nn=(endep(end)-stdep(1))/rlev+1;
dep=stdep(1):rlev:endep(end);


type1='TYP1, TYP2, TYP3, TYP4, TYP5, TYP6, TYP7, TYP8, TYP9, TY10, TY11, TY12, TY13, TY14, TY15, TY16, TY17, TY18, TY19, TY20, TYPC'; 
type2='DEPTH       TYP1    TYP2     TYP3     TYP4     TYP5     TYP6     TYP7     TYP8     TYP9     TY10';        
type3='TY11     TY12     TY13     TY14     TY15     TY16     TY17     TY18     TY19     TY20     TYPC';
fprintf(fp,'%s\n',type1);
fprintf(fp,'%s   %s\n',type2,type3);

for i=1:1:nn
    for j=1:1:20
        TYPE1(i,j)=0;
    end
end
for i=1:1:nn
    for j=1:1:length(stdep)
        if dep(i)>=stdep(j)&&dep(i)<endep(j)
            TYPE1(i,predict_labels(j))=predict_labels(j);
        end
    end
end

typc=sum(TYPE1')';
TYPE2(:,1)=dep;
TYPE2(:,2:21)=TYPE1;
TYPE2(:,22)=typc;


for i=1:1:nn
    fprintf(fp,'%7.3f   %6.3f   %6.3f   %6.3f   %6.3f    %6.3f   %6.3f   %6.3f   %6.3f   %6.3f   %6.3f  %6.3f   %6.3f   %6.3f   %6.3f   %6.3f   %6.3f   %6.3f   %6.3f   %6.3f   %6.3f   %6.3f\n',TYPE2(i,:));
end


fclose(fp);



end


