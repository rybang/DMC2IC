function [vecResult,bestC1] = DMC2IC_choose_C1(trainX,trainY,testX,testY,C2,opt)
vecResult = [];
C1base = opt.C1base;
C1range = opt.C1range;
C1length = length(C1range);
opt.C2 = C2;
for C_w = 1:1:C1length % find the best C1
    C1 = C1base^C1range(C_w); 
    opt.C1 = C1;
    svmDML = DMC2IC_GBCD(trainX,trainY,opt);
    ResultC1 = DMC2IC_test(svmDML.svm,testX,testY);
    vecResult = [vecResult,ResultC1.accuracy];
    disp( strcat( 'svmDML test with C1:',num2str( C1base ),'^',num2str( C1range(C_w) ),'  C2:',num2str(C2) ,'  itr==',num2str( svmDML.itr ) ));
    disp( strcat('C1 accuracy:',num2str(ResultC1.accuracy),'...'));
    disp(strcat('train time:',num2str(svmDML.trainTime),'...'));
    fprintf('\n')
end
id = find( vecResult == max(vecResult) );
bestC1 = C1base^C1range( id(1) );
end