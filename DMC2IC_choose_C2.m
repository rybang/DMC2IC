function [vecResult,bestC2] = DMC2IC_choose_C2(trainX,trainY,testX,testY,C1,opt)
vecResult = [];
C2base = opt.C2base;
C2range = opt.C2range;
C2length = length(C2range);
opt.C1 = C1;
for C_M = 1:1:C2length % find the best C2
    C2 = C2base^C2range(C_M);
    opt.C2 = C2;
    svmDML = DMC2IC_GBCD(trainX,trainY,opt);
    ResultC2 = DMC2IC_test(svmDML.svm,testX,testY);
    vecResult = [vecResult,ResultC2.accuracy];
    disp( strcat( 'svmDML test with C1:',num2str(C1),'  C2:',num2str( C2base ),'^',num2str( C2range(C_M) ) ,'  itr==',num2str( svmDML.itr ) ));
    disp( strcat('C2 accuracy:',num2str(ResultC2.accuracy),'...'));
    disp(strcat('train time:',num2str(svmDML.trainTime),'...'));
    fprintf('\n')
end
id = find( vecResult == max(vecResult) );
bestC2 = C2base^C2range( id(1) );
end