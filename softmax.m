function S = softmax(theta, X)
%function S = softmax(theta, X)
%
% 

K = length(theta);
N = size(X,1);

F = zeros(N,K);
for k=1:K
    F(:,k) = X*theta{k}; 
end

ma = max(F,[],2);
Fma = F - ma;
expF = exp(Fma);
S = expF ./ sum(expF, 2);

