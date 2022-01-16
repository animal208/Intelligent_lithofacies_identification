%% 数据归一化处理
% 作者：weiwu dong                     2020.12.7
% 对矩阵data的“各列”进行归一化处理
% 参数说明：
%         1、data、data_scale：归一化前后的数据矩阵；
%         2、sign=1，自动选取各列最大、最小值；
%            sign=2，人为指定各列最大、最小值,每列的最大、
%                    最小值由maxdww和mindww数组依次给出；
%         3、数据归一化后范围：[ymin,ymax].




function [data_scale]=scale_dww(data,sign,ymin,ymax,mindww,maxdww)


%%
if sign==1
    data_min=min(data);
    data_max=max(data);
elseif sign==2
    data_min=mindww;
    data_max=maxdww;
else
    error('sign error!!!!\n')
end

%%
[n,m]=size(data);
for j=1:1:m
    for i=1:1:n
        data_temp(i,j)=(data(i,j)-data_min(j))/(data_max(j)-data_min(j));
        if data_temp(i,j)<0
           data_temp(i,j)=0;
        end
        if data_temp(i,j)>1
           data_temp(i,j)=1;
        end
        
        data_scale(i,j)=(ymax-ymin)*data_temp(i,j)+ymin;
        if data_scale(i,j)<ymin
           data_scale(i,j)=ymin;
        end
        if data_scale(i,j)>ymax
           data_scale(i,j)=ymax;
        end
    end 
end


end





