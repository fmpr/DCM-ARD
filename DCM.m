function S = DCM(theta, X, availableChoices)
%function S = DCM(theta, X, availableChoices)
%
% (C) Filipe Rodrigues (2019) 

K = length(theta);
N = size(X{1},1);

F = zeros(N,K);
for k=1:K
    F(:,k) = X{k}*theta{k}; 
end

ma = max(F,[],2);
Fma = F - ma;
expF = exp(Fma);
expF = expF .* availableChoices; % assign probability zero for unavailable alternatives
S = expF ./ sum(expF, 2);

