function [F, mu, C] = dsvi_DCM_FullCov(mu, C, loglik, logprior, options)
%function [F, mu, C] = dsvi_DCM_FullCov(mu, C, loglik, logprior, options)
%
% What it does: applies doubly stochastic variational inference 
% in a Discrete Choice Model (DCM).
%
% Inputs 
%         - mu: D x 1 mean vector of the variational distribution.  
%         - C: scale matrix of the variational distribution.
%              If the size is D x D, then a full lower triangular positive 
%              definite (Cholesky) matrix is learned.  
%              If the size is D x 1, then a fully factorized approximation is
%              learned. 
%         - loglik: a structure containing a function handle and the input arguments for
%            the log likelihood. 
%         - logprior: a structure containing a function handle and the input arguments for
%            the log prior over the parameters.   
%         - options: 
%                options(1) is the number of stochastic approximation iterations.  
%                options(2) is the fixed learning rate for mu (while
%                            0.1*options(2) is the corresponding rate for C).
%                options(3) is the ratio between the full length of the dataset and the size of the minibatch
%                           (if training is done with a full dataset this is just 1) 
%                options(4) if 1, it uses as the standardized distribution the standard normal. 
%                           if 2, it uses a product of standard logistic distributions. 
%
% Outputs   
%         - F: a vector with all stochastic instantaneous values of the
%              lower bound. 
%         - mu: the final/learned value for mu. 
%         - C:  the final/learned value for C. 
% 
%    
% Filipe Rodrigues (2019)
% Based on the original implementation of Michalis Titsias for binary logistic regression (2014)

K = length(mu);     % number of classes
D = zeros(K,1);
for k=1:K
    D(k) = length(mu{k});
end
[D1, D2] = size(C{1});

if options(3) == 0
    options(3) = 1;
end

% Ratio between the full length of the dataset and the minibatch
% This simple will be eqaul to 1 if all the data are used 
Nn = options(3); 

if options(4) == 0
   options(4) = 1;
end
whichStandDist = options(4);  

if D2 == 1
    diagfull = 1; 
elseif D2 == D1
    diagfull = 0;
    tmpC = cell(3,1);
    for k=1:K
        tmpC{k} = triu(ones(D(k)))';
    end
else
    error('Something is wrong with the initial C: must be either D x D or D x 1.')
end

Niter = options(1); % Number of likelihood/gradient evaluations
ro = options(2) ;   % Learning rate

F = zeros(1,Niter);
N = size(loglik.inargs{2},1);
minibatch_size = ceil(N/Nn);
perm = randperm(N);
z = cell(3,1);
theta = cell(3,1);
g = cell(3,1);
dg = cell(3,1);
dmu = cell(3,1);
dC = cell(3,1);
for n = 1:Niter
%
    for k=1:K
        if whichStandDist == 1      % Gaussian 
            z{k} = randn(D(k),1);   
        elseif whichStandDist == 2  % Logistic distribution
            z{k} = rand(D(k),1);   
            z{k} = log(z{k}./(1-z{k}));
        end
        
        if diagfull == 1
            theta{k} = C{k}.*z{k} + mu{k};
        else
            theta{k} = C{k}*z{k} + mu{k};
        end
    end
    
    minibatch_no = mod(n, Nn);
    lo = minibatch_no*minibatch_size+1;
    hi = min((minibatch_no+1)*minibatch_size, N);
    [g_lik, dg_lik] = loglik.name(theta, loglik.inargs{:}, perm(lo:hi));
    
    g = Nn*g_lik;
    logdetC = 0.0;
    for k=1:K
        [g_prior, dg_prior] = logprior.name(theta{k}, logprior.inargs{k}{:});
        g = g + g_prior;
    
        % Stochastic gradient wrt (mu,C) of the lower bound 
        dg{k} = Nn*dg_lik{k} + dg_prior;
        
        if diagfull == 1 
           dmu{k} = dg{k};
           dC{k} = (dg{k}.*z{k}) + 1./C{k};
        else     
           dmu{k} = dg{k};
           dC{k} = (dg{k}*z{k}').*tmpC{k} + diag(1./diag(C{k}));
        end
        
        % Update the variational parameters
        mu{k} = mu{k} + ro*dmu{k}; 
        C{k} = C{k} + (0.1*ro)*dC{k};
    
        if diagfull == 1
           C{k}(C{k}<=1e-4)=1e-4;              % constraint (for numerical stability and positive definitenes)  
           logdetC = logdetC + sum(log(C{k}));
        else 
           keep = diag(C{k});
           keep(keep<=1e-4)=1e-4;        % constraint (for numerical stability and positive definitenes)
           C{k} = C{k} + (diag(keep - diag(C{k})));
           logdetC = logdetC + sum(log(diag(C{k})));
        end
    end
    
    % entropy of the standardized distribution
    if whichStandDist == 1
       entr = 0.5*D(k) + 0.5*D(k)*log(2*pi);  % Gaussian 
    elseif whichStandDist == 2         
       entr = D(k)*2;                      % product of lostistics
    end
    
    % stochastic value of the lower bound 
    F(n) = g + logdetC + entr;
%    
end
