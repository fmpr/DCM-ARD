clear;
close all;
rng(1); % fix seed for reproducibility

inputFile = '../PyLogit/V2/swissmetro.dat';

fprintf('Input file: %s\n', inputFile);

% load data
Mtable = readtable(inputFile);
M = table2array(Mtable);

% remove data with choice == 0 (missing)
M(M(:,28) == 0,:) = [];

% remove data with age == 6 (other)
M(M(:,10) == 6,:) = [];

% remove data with purpose == 9 (other)
M(M(:,5) == 9,:) = [];
    
% shuffle data completely at random
%M_shuffled = M(randperm(size(M,1)),:);

% shuffle data accounting for individual IDs
unique_ids = unique(M(:,4));
shuffled_ids = unique_ids(randperm(length(unique_ids)),:);
M_shuffled = zeros(size(M));
for i=1:length(unique_ids)
    id = shuffled_ids(i);
    lo = (i-1)*9+1;
    hi = i*9;
    M_shuffled(lo:hi,:) = M(M(:,4)==id,:);
end

csvwrite('swissmetro_shuffled_individuals.csv', M_shuffled)