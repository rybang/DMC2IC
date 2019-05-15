%objective value of the dual SVM problem
function objValue = calObjValue(X,Y,XNNs,XNNd,svm,dml,C1,C2,inv_eps,opt)
t1=clock;
Label = svm.Label;
wM = svm.wM;
M = dml.M;
r = dml.r;
invM = inverseM(M,inv_eps);
instanceNum = size(X,1);
tmp_GM = 0;
for i_GM=1:1:size(wM,1)
    tmp_GM = tmp_GM + (wM(i_GM,:) * invM * wM(i_GM,:)');
end
objValue = 0.5 * tmp_GM + 0.5 * vec(M)' * vec(M);
if opt.isKernel ==1
    objValue = 0.5 * tmp_GM + 0.5 * trace((X*M)^2);
end
sai = 0;
score = X*wM';
for i=1:1:size(score,1)
    j = find(Label==Y(i));
    sc_belong = score(i,j);
    pos_not_belong = find(Label~=Y(i));
    temp_sc_row = score(i,pos_not_belong);
    sc_not_belong_max = max(temp_sc_row);
    temp_sai = 1 - sc_belong + sc_not_belong_max;
    if(temp_sai > 0 )
        sai = sai + temp_sai;
    end
end
objValue = objValue + C1*sai;

eta_s = 0;
for i = 1:1:instanceNum
    dist_s = XNNs(i,:)*M*XNNs(i,:)';
    if dist_s > r-1
        eta_s = eta_s + dist_s -r + 1;
    end
end

eta_d = 0;
for i=1:1:instanceNum
    dist_d = XNNd(i,:)*M*XNNd(i,:)';
    if dist_d < r+1
        eta_d = eta_d + r + 1 - dist_d;
    end
end
eta = eta_s+eta_d;
objValue = objValue + C2*eta;
t2=clock;
calObjTime=etime(t2,t1);
%disp(strcat('total calObjTime time:',num2str(calObjTime),'s'));
end