%     SVM_Model_for_Lithofacies_Identification
%    Author:weiwu dong        date: 2021/11/12




% read the input data

fp=fopen('SVM_class_train.TXT','r');
i=1;
while feof(fp)==0
    temp=fgetl(fp);
    data_sample(i,:)=strread(temp,'%f');
    i=i+1;
end
fclose(fp);

data_sample(:,3)=fix(data_sample(:,3));
labels=data_sample(:,3);
data=data_sample(:,4:end);   %样本集
[temp n_vector]=size(data);



fp=fopen('SVM_class_predict.TXT','r');
i=1;
while feof(fp)==0
    temp=fgetl(fp);
    fulldata_predict(i,:)=strread(temp,'%f');
    i=i+1;
end
fclose(fp);

stdep=fulldata_predict(:,1);
endep=fulldata_predict(:,2);
data_predict=fulldata_predict(:,4:end);%待测数据集


filename_predict_TYPE=['SVM_class_predict_TYPE.txt'];

sign_predict_file=1;


% parameter settings
sign_scale=1;
min_scale=0;
max_scale=1;
sign_data=1;

sign_scale_save=0;
sign_drm=3;
sign_drm_save=0;
sign_rescale=0;

sign_data=1;
k_fold=5;
sign_data_save=0;

mysvm.svm_type=0;
mysvm.kernel_type=2;
mysvm.degree=3;
mysvm.coef0=0;
mysvm.nu=0.5;
mysvm.epsilon=0.1;
mysvm.cachesize=100;
mysvm.eps=0.001;
mysvm.weight=1;
mysvm.shrinking=1;
mysvm.probability_estimates=0;
 sign_pom=1;
 
mynet.cmin=-10;
mynet.cmax=15;
mynet.gmin=-10;
mynet.gmax=15;
mynet.v=5;
mynet.cstep=1;
mynet.gstep=1;
mynet.accstep=4.5;

cost_dww=32;
gamma_dww=32;



[model,TYPE]=SVM_Class_Func(labels,data,stdep,endep,data_predict,...
    sign_scale,sign_scale_save,min_scale,max_scale,[],[],sign_drm,sign_drm_save,sign_pom,0,...
    sign_data,sign_data_save,k_fold,sign_rescale,mynet,[],[],cost_dww,gamma_dww,filename_predict_TYPE,mysvm);



for i=1:1:100
    str(i,:)='                                                                       ';
end

fp=fopen('out.txt','r');
i=1;
while feof(fp)==0
    temp=fgetl(fp);
    kk=length(temp);
    str(i,1:kk)=temp;
    fprintf('%s\n',str(i,1:kk));
    i=i+1;
end


fclose(fp);



