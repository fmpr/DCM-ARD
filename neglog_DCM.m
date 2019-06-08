function [g, dg] = neglog_DCM(theta, X, Y, T, availableChoices)
%function [g, dg] = neglog_DCM(theta, X, Y, T, availableChoices)
%
% (C) Filipe Rodrigues (2019) 

[N,K] = size(T);

F = zeros(N,K);
for k=1:K
    F(:,k) = X{k}*theta{k}; 
end

ma = max(F,[],2);
Fma = F - ma;
expF = exp(Fma);
expF = expF .* availableChoices; % assign probability zero for unavailable alternatives
normExpF = sum(expF, 2);
S = expF ./ normExpF;

Yind = sub2ind(size(Fma), [1:N]', Y);
g = -sum(Fma(Yind) - log(normExpF));
dg = cell(3,1);
for k=1:K
    dg{k} = -X{k}'*(T(:,k)-S(:,k));
end
