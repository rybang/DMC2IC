%Xt n*d matrix
%Yt n*1 vector
function result = DMC2IC_test(svm, X, Y)
Label = svm.Label;
Xt = X; 
Yt = zeros(length(Y),1);
score = Xt*svm.wM';
result.score = score;
for i=1:1:size(score,1)
    sc_tmp = score(i,:);
    sc_max = max(sc_tmp);
    pos_sc_max = find(sc_tmp == sc_max);
    Yt(i) = Label(pos_sc_max(1));
end
result.Y = Yt;
result.accuracy = size(find(Yt==Y))/size(Yt);  % Ô¤²â¾«¶È
