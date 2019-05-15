%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function svm = DMC2IC_svm(X,Y,dml,C1) % X是训练样本，Y是训练样本的标签
svm = struct();
LXr_sparse = sparse(X*dml.L');
liblinear_opt = ['-s 4 -q -c ',num2str(C1)];
liblinear_model = train(Y, LXr_sparse, liblinear_opt);
wL = liblinear_model.w;
svm.wM = wL*dml.L;
svm.Label =  liblinear_model.Label;
svm.model = liblinear_model;
end