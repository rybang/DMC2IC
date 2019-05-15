function invM=inverseM(M,inv_eps)
I = eye(size(M));
pM = inv_eps*I;
invM = (M+pM)^-1;
end