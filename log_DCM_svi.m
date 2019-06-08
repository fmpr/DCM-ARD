function [g, dg] = log_DCM_svi(theta, X, Y, T, availableChoices, batchIdx)
%function [g, dg] = log_DCM_svi(theta, X, Y, T, availableChoices, batchIdx)
%
% (C) Filipe Rodrigues (2019)

[~,K] = size(T);
N = length(batchIdx);

F = zeros(N,K);
for k=1:K
    F(:,k) = X{k}(batchIdx,:)*theta{k}; 
end

ma = max(F,[],2);
Fma = F - ma;
expF = exp(Fma);
expF = expF .* availableChoices(batchIdx,:); % assign probability zero for unavailable alternatives
normExpF = sum(expF, 2);
S = expF ./ normExpF;

Yind = sub2ind(size(Fma), [1:N]', Y(batchIdx));
g = sum(Fma(Yind) - log(normExpF));
dg = cell(3,1);
for k=1:K
    dg{k} = X{k}(batchIdx,:)'*(T(batchIdx,k)-S(:,k));
end
