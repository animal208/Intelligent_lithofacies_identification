function [bestacc,bestc,bestg] = SVMcgForClass(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep,svm_option)


%% about the parameters of SVMcg 
if nargin < 10
    accstep = 4.5;
end
if nargin < 8
    cstep = 0.8;
    gstep = 0.8;
end
if nargin < 7
    v = 5;
end
if nargin < 5
    gmax = 8;
    gmin = -8;
end
if nargin < 3
    cmax = 8;
    cmin = -8;
end
%% X:c Y:g cg:CVaccuracy
[X,Y] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);
[m,n] = size(X);
cg = zeros(m,n);
eps = 1e-1;
%% record acc with different c & g,and find the bestacc with the smallest c
bestc = 1;
bestg = 0.1;
bestacc = 0;
basenum = 2;

svm_type=num2str(svm_option.svm_type);
kernel_type=num2str(svm_option.kernel_type);
degree=num2str(svm_option.degree);
coef0=num2str(svm_option.coef0);
nu=num2str(svm_option.nu);
epsilon=num2str(svm_option.epsilon);
cachesize=num2str(svm_option.cachesize);
eps2=num2str(svm_option.eps);
weight=num2str(svm_option.weight);
shrinking=num2str(svm_option.shrinking);
probability_estimates=num2str(svm_option.probability_estimates);
    
for i = 1:m
    for j = 1:n
         cmd = ['-v ',num2str(v),' -c ',num2str( basenum^X(i,j) ),' -g ',num2str( basenum^Y(i,j) ),' -s ',svm_type,' -t ',kernel_type,' -d ',degree,' -r ',coef0,' -n ',nu,' -p ',epsilon,' -m ',cachesize,' -e ',eps2,' -wi ',weight,' -h ',shrinking,' -b ',probability_estimates];
%         cmd = ['-v ',num2str(v),' -c ',num2str( basenum^X(i,j) ),' -g ',num2str( basenum^Y(i,j) ),' -s ',svm_type,' -t ',kernel_type];
%        cmd = ['-v ',num2str(v),' -c ',num2str( basenum^X(i,j) ),' -g ',num2str( basenum^Y(i,j) )];
        cg(i,j) = svmtrain(train_label, train, cmd);
        
        if cg(i,j) <= 55
            continue;
        end
        
        if cg(i,j) > bestacc
            bestacc = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end        
        
        if abs( cg(i,j)-bestacc )<=eps && bestc > basenum^X(i,j) 
            bestacc = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end        
        
    end
end
%% to draw the acc with different c & g
figure;
[C,h] = contour(X,Y,cg,70:accstep:100);
clabel(C,h,'Color','r');
xlabel('log2c','FontSize',12);
ylabel('log2g','FontSize',12);
firstline = 'SVC参数选择结果图(等高线图)[GridSearchMethod]'; 
secondline = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
    ' CVAccuracy=',num2str(bestacc),'%'];
title({firstline;secondline},'Fontsize',12);
grid on; 

figure;
meshc(X,Y,cg);
% mesh(X,Y,cg);
% surf(X,Y,cg);
axis([cmin,cmax,gmin,gmax,30,100]);
xlabel('log2c','FontSize',12);
ylabel('log2g','FontSize',12);
zlabel('Accuracy(%)','FontSize',12);
firstline = 'SVC参数选择结果图(3D视图)[GridSearchMethod]'; 
secondline = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
    ' CVAccuracy=',num2str(bestacc),'%'];
title({firstline;secondline},'Fontsize',12);