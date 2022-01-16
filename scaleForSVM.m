function [train_scale,test_scale,ps] = scaleForSVM(train_data,test_data,ymin,ymax)

%%
if nargin < 3
    ymin = 0;
    ymax = 1;
end
if nargin < 2
    test_data = train_data;
end
%%
[mtrain,ntrain] = size(train_data);
[mtest,ntest] = size(test_data);

dataset = [train_data;test_data];

[dataset_scale,ps] = mapminmax(dataset',ymin,ymax);

dataset_scale = dataset_scale';
train_scale = dataset_scale(1:mtrain,:);
test_scale = dataset_scale( (mtrain+1):(mtrain+mtest),: );
