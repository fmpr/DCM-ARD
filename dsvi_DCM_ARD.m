function [F, mu, C] = dsvi_DCM_ARD(mu, C, loglik, options, groups, Dgroup)
%function [F, mu, C] = dsvi_DCM_ARD(mu, C, loglik, options, groups, Dgroup)
%
% Inputs 
%         - mu: D x 1 mean vector of the variational distribution.  
%         - C:  D x 1 scale vector associated with a fully factorized approximation. 
%         - loglik: a structure containing a function handle and the input arguments for
%            the log likelihood. 
%         - options: 
%                options(1) is the number of stochastic approximation iterations.  
%                options(2) is the fixed learning rate for mu (while
%                            0.1*options(2) is the corresponding rate for C).
%                options(3) is the ratio between the full length of the dataset and the size of the minibatch
%                           (if training is done with a full dataset this is just 1) 
%         - groups: cell array with assignment of input dimensions to
%             variable groups
%         - Dgroups: cell array with dimensions (D_g) for each group of
%             variables
%
% Outputs   
%         - F: a vector with all stochastic instantaneous values of the
%              lower bound. 
%         - mu: the final/learned value for mu. 
%         - C:  the final/learned value for C. 
%
% Filipe Rodrigues (2019)
% Based on the original implementation of Michalis Titsias for binary logistic regression (2014)


K = length(mu);     % number of classes
D = zeros(K,1);
for k=1:K
    D(k) = length(mu{k});
end

if options(3) == 0
    options(3) = 1;
end

% Ratio between the full length of the dataset and the minibatch
% This simple will be equal to 1 if all the data are used 
Nn = options(3); 

Niter = options(1); % Number of likelihood/gradient evaluations
ro = options(2) ;   % Learning rate

F = zeros(1,Niter);
N = size(loglik.inargs{2},1);
minibatch_size = ceil(N/Nn);
perm = randperm(N);
z = cell(3,1);
theta = cell(3,1);
dg = cell(3,1);
for n = 1:Niter
    for k=1:K
        z{k} = randn(D(k),1);   
        theta{k} = C{k}.*z{k} + mu{k};
    end
    
    minibatch_no = mod(n, Nn);
    lo = minibatch_no*minibatch_size+1;
    hi = min((minibatch_no+1)*minibatch_size, N);
    [g_lik, dg_lik] = loglik.name(theta, loglik.inargs{:}, perm(lo:hi));

    for k=1:K
        dg{k} = Nn*dg_lik{k};
    end

    % stochastic value of the lower bound:
    % data term plus the optimal KL term (added later in the for loop)
    F(n) = Nn*g_lik;
    
    for k=1:K
        C2 = C{k}.*C{k};
        
        % Cmu = C2 + mu{k}.^2;
        Cmu = zeros(D(k),1);
        for d=1:(D(k))
            g = groups{k}(d);
            Cmu(d) = sum(C{k}(groups{k}==g).^2 + mu{k}(groups{k}==g).^2);
        end

        % Stochastic gradient update of the parameters
        dmu = dg{k} - (Dgroup{k}.*mu{k})./Cmu;
        dC = (dg{k}.*z{k}) + 1./C{k} - (Dgroup{k}.*C{k})./Cmu;

        mu{k} = mu{k} + ro*dmu; 
        C{k} = C{k} + (0.1*ro)*dC;

        C{k}(C{k}<=1e-4)=1e-4;      % constraint (for numerical stability)  
        
        % add optimal KL term to the stochastic value of the lower bound
        %F(n) = F(n) + 0.5*sum(log(C2./Cmu)); 
        F(n) = F(n) + 0.5*sum(log((Dgroup{k}.*C2)./Cmu)); 
    end
%    
end
 
