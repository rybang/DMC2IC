
function svmDML = DMC2IC_GBCD(trainX,trainY,opt)

C1 = opt.C1;
C2 = opt.C2;
maxStopItr = opt.maxStopItr;
setpsize = opt.setpsize;
psd_eps = opt.psd_eps;
con_eps = opt.con_eps; 
inv_eps = opt.inv_eps; 
factor = opt.factor;
vecobj = [];
X = trainX;
Y = trainY;

M0 = opt.M0;
[psdM,psdL] = makepsd(M0,psd_eps);
dml.L=psdL;
dml.M=psdM;
dml.r = opt.r;
itr = opt.itrOptNum;
XNNs = opt.trainXNNs; 
XNNd = opt.trainXNNd; 
NN = opt.NN;

svm = DMC2IC_svm(X,Y,dml,C1);

t1=clock;
stopItr = 0;
instanceNum = size(X,1);
for i=1:1:itr
    
    %fix wM to train M
    Mt = dml.M;
    rt = dml.r;
    

    GM_s = zeros(size(Mt));
    Gr_s = 0;
    for ins_i = 1:1:instanceNum
        if XNNs(ins_i,:)*Mt*XNNs(ins_i,:)' <= rt-1
            continue;
        end

        GM_s = GM_s + NN(ins_i).Matrix_XNNs;
        Gr_s = Gr_s + 1;
    end
    GM_s = C2*GM_s;
    Gr_s = C2*Gr_s;

    GM_d = zeros(size(Mt));
    Gr_d = 0;
    for ins_i=1:1:instanceNum
        if XNNd(ins_i,:)*Mt*XNNd(ins_i,:)' >= rt+1
            continue;
        end
        GM_d = GM_d + NN(ins_i).Matrix_XNNd;
        Gr_d = Gr_d + 1;
    end
    GM_d = C2*GM_d;
    Gr_d = C2*Gr_d;

    wM = svm.wM;
    invM = inverseM(Mt,inv_eps);
    tmp_GM = 0;
    for i_GM=1:1:size(wM,1)
        A = wM(i_GM,:)'*wM(i_GM,:);
        tmp_GM = tmp_GM + (invM*A*invM);
    end
    GM = -0.5 * tmp_GM + Mt + GM_s - GM_d;
    if opt.isKernel ==1
        GM = -0.5 * tmp_GM + X*Mt*X + GM_s - GM_d;
    end
    Gr = -Gr_s + Gr_d;

    tempM = Mt - setpsize*GM;
    tempr = rt - setpsize*Gr;

    [psdM,psdL] = makepsd(tempM,psd_eps);
    dml.L=psdL;
    dml.M=psdM;
    dml.r = tempr;
    
    %fix M to train wM
    svm = DMC2IC_svm(X,Y,dml,C1);

    objValue = calObjValue(X,Y,XNNs,XNNd,svm,dml,C1,C2,inv_eps,opt);
    vecobj = [vecobj,objValue];
    if(i>1)
        if abs( vecobj(i)-vecobj(i-1) ) < con_eps 
            stopItr = stopItr + 1;
        else
            stopItr = 0;
        end
        if vecobj(i)-vecobj(i-1) > 0 
            dml = old_dml;
            svm = old_svm;
            break;
        end
    end
    if maxStopItr == stopItr
        break;
    end
    old_dml = dml;
    old_svm = svm;
    setpsize = setpsize * factor;
end
svmDML.vecobj = vecobj; 
svmDML.stopItr = stopItr;
svmDML.itr = i; 
t2=clock;
trainTime=etime(t2,t1);
svmDML.svm = svm;
svmDML.dml = dml;
svmDML.trainTime = trainTime;
end
