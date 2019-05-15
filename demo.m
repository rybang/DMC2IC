clear;
close all;
%parpool('local',4);
load glass-test.mat

fprintf('\ntraining the svmDML classifier.\n');
%opt.mc = mc;
C1base = 2;
C1range = -5:1:5;

C2base = 10;
C2range = -5:1:2;
initC2 = C2base^0;
opt.setpsize = 1e-2; % selected from {1e-0,1e-1,1e-2,1e-3,1e-4,1e-5}

opt.isKernel = 1; % 0 is non-kernel, 1 is kernel
opt.kernelType = 'rbf_fast';
opt.delta = 1e-1;

opt.C1base = C1base;
opt.C1range = C1range;
opt.C2base = C2base;
opt.C2range = C2range;
opt.itrOptNum = 15;%
opt.maxStopItr = 3;
opt.psd_eps = 1e-10; 
opt.con_eps = 1e-1; 
opt.inv_eps = 1e-8; 
opt.factor = 0.9*1.01;

fprintf('\n')
trainX_original = trainX;

if opt.isKernel ==1
    trainX = kernel_svmDML(trainX,trainX_original',opt);
    testX = kernel_svmDML(testX,trainX_original',opt);
    opt.KtrainX = trainX;
end


tempM0 = eye(size(trainX,2));
tempr = 1;
opt.M0 = tempM0;
opt.r = tempr;
opt.fastSearchNN = 1;
t1=clock;
[trainXNNs,trainXNNd,NN] = SearchNN(trainX,trainY,opt);
opt.trainXNNs = trainXNNs; 
opt.trainXNNd = trainXNNd; 
opt.NN = NN;
t2=clock;
NNTime2=etime(t2,t1);


% find the best C1
[vecResult2,bestC1] = DMC2IC_choose_C1(trainX,trainY,testX,testY,initC2,opt);
fprintf('..\n')
% find the best C2
[vecResult1,bestC2] = DMC2IC_choose_C2(trainX,trainY,testX,testY,bestC1,opt);
fprintf('.\n')

%final training with the best parameters
opt.C1 = bestC1;
opt.C2 = bestC2;
t1=clock;
svmDML = DMC2IC_GBCD(trainX,trainY,opt);
t2=clock;
trainTime=etime(t2,t1);

finalResult = DMC2IC_test(svmDML.svm,testX,testY);
model = svmDML.svm.model;
model.w = svmDML.svm.wM;
[predict_label, accuracy, dec_values] = predict(testY, sparse(testX), model);

disp(strcat( 'svmDML test with bestC1:',num2str(bestC1),'  bestC2:',num2str(bestC2) ));
disp(strcat('final accuracy:',num2str(finalResult.accuracy),'...'));
disp(strcat('final training time:',num2str(trainTime),'...'));
%keyboard;
