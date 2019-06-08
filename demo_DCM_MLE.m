clear;
close all;
rng(1); % fix seed for reproducibility

outdir = 'output_dcm_mle_v2/';
inputFile = 'swissmetro_processed.csv';
dataName = 'swissmetro_varalt_shuffled_groups_easytrue_spec1';

fprintf('Output directory: %s\n', outdir);
fprintf('Input file: %s\n', inputFile);
fprintf('Data name: %s\n', dataName);

% load data
M = csvread(inputFile,1);

% remove data with purpose == 9 (other)
M(M(:,5) == 9,:) = [];
    
% shuffle data completely at random
%M_shuffled = M(randperm(size(M,1)),:);

% shuffle data accounting for individual IDs
unique_ids = unique(M(:,1));
shuffled_ids = unique_ids(randperm(length(unique_ids)),:);
M_shuffled = zeros(size(M));
for i=1:length(unique_ids)
    id = shuffled_ids(i);
    lo = (i-1)*9+1;
    hi = i*9;
    M_shuffled(lo:hi,:) = M(M(:,1)==id,:);
end

% process data
ids = M_shuffled(:,1);
Y_true = M_shuffled(:,2);
age = M_shuffled(:,3);
ga = M_shuffled(:,4)+1;
purpose = M_shuffled(:,5);
new_purpose = -1*ones(size(purpose));
new_purpose((purpose == 1) | (purpose == 3) | (purpose == 5) | (purpose == 7)) = 1;
new_purpose((purpose == 2) | (purpose == 6)) = 2;
new_purpose((purpose == 4) | (purpose == 8)) = 3;
purpose = new_purpose;
who = M_shuffled(:,6)+1;
luggage = M_shuffled(:,7)+1; 
luggage(luggage == 4) = 3;
income = M_shuffled(:,8)+1;
availableChoices = M_shuffled(:,9:11);
TRAIN_ASC = ones(length(M),1);
TRAIN_TT = M_shuffled(:,12)/60.0;
TRAIN_CO = M_shuffled(:,15)*0.01;
TRAIN_HE = M_shuffled(:,18)/60.0;
SM_ASC = ones(length(M),1);
SM_TT = M_shuffled(:,13)/60.0;
SM_CO = M_shuffled(:,16)*0.01;
SM_HE = M_shuffled(:,19)/60.0;
CAR_ASC = ones(length(M),1);
CAR_TT = M_shuffled(:,14)/60.0;
CAR_CO = M_shuffled(:,17)*0.01;

% define specifications to test
alternative_names = {'TRAIN', 'SM', 'CAR'};

% DCM specification
fitSpec = cell(length(alternative_names),1);

% spec1: super-simple baseline DCM model
fitSpec{1} = {'TRAIN_ASC', 'TRAIN_TT', 'TRAIN_CO'};
fitSpec{2} = {'SM_ASC', 'SM_TT', 'SM_CO'};
fitSpec{3} = {'CAR_TT', 'CAR_CO'};

% % spec2: very simplified version of spec given by ARD
% fitSpec{1} = {'TRAIN_ASC',  'logt(TRAIN_TT)', 'logt(TRAIN_TT) x ga', 'logt(TRAIN_CO)'};
% fitSpec{2} = {'SM_ASC', 'logt(SM_TT)', 'logt(SM_CO)'};
% fitSpec{3} = {'CAR_TT', 'logt(CAR_CO)'};

% % spec3: simplified version of spec given by ARD
% fitSpec{1} = {'TRAIN_ASC', 'logt(TRAIN_TT)', 'logt(TRAIN_TT) x ga', 'logt(TRAIN_CO)', 'logt(TRAIN_CO) x purpose'};
% fitSpec{2} = {'SM_ASC', 'logt(SM_TT)', 'logt(SM_CO)', 'logt(SM_CO) x ga'};
% fitSpec{3} = {'CAR_TT', 'logt(CAR_CO)'};

% % spec4: more complex version of spec given by ARD
% fitSpec{1} = {'TRAIN_ASC', 'logt(TRAIN_TT)', 'logt(TRAIN_TT) x ga', 'logt(TRAIN_CO)', 'logt(TRAIN_CO) x ga', 'logt(TRAIN_CO) x purpose'};
% fitSpec{2} = {'SM_ASC', 'logt(SM_TT)', 'logt(SM_CO)', 'logt(SM_CO) x ga', 'logt(SM_CO) x purpose'};
% fitSpec{3} = {'CAR_TT', 'CAR_TT x ga', 'logt(CAR_CO)'};

% % spec5: even more complex version of spec given by ARD
% fitSpec{1} = {'TRAIN_ASC', 'logt(TRAIN_TT)', 'logt(TRAIN_TT) x ga', 'logt(TRAIN_CO)', 'logt(TRAIN_CO) x ga', 'logt(TRAIN_CO) x purpose', 'logt(TRAIN_CO) x age'};
% fitSpec{2} = {'SM_ASC', 'SM_ASC x age', 'logt(SM_TT)', 'logt(SM_CO)', 'logt(SM_CO) x ga', 'logt(SM_CO) x purpose'};
% fitSpec{3} = {'CAR_TT', 'CAR_TT x ga', 'CAR_TT x purpose', 'logt(CAR_CO)'};

% % spec6: almost full version of spec given by ARD
% fitSpec{1} = {'TRAIN_ASC', 'logt(TRAIN_TT)', 'logt(TRAIN_TT) x ga', 'logt(TRAIN_CO)', 'logt(TRAIN_CO) x ga', 'logt(TRAIN_CO) x purpose', 'logt(TRAIN_CO) x age', 'logt(TRAIN_HE)'};
% fitSpec{2} = {'SM_ASC', 'SM_ASC x age', 'logt(SM_TT)', 'logt(SM_CO)', 'logt(SM_CO) x ga', 'logt(SM_CO) x purpose'};
% fitSpec{3} = {'CAR_TT', 'CAR_TT x ga', 'CAR_TT x purpose', 'logt(CAR_CO)'};

% % spec7: full version of spec given by ARD
% fitSpec{1} = {'TRAIN_ASC', 'logt(TRAIN_TT)', 'logt(TRAIN_TT) x ga', 'logt(TRAIN_CO)', 'logt(TRAIN_CO) x ga', 'logt(TRAIN_CO) x purpose', 'logt(TRAIN_CO) x age', 'logt(TRAIN_HE)'};
% fitSpec{2} = {'SM_ASC', 'SM_ASC x ga', 'SM_ASC x age', 'logt(SM_TT)', 'logt(SM_CO)', 'logt(SM_CO) x ga', 'logt(SM_CO) x purpose'};
% fitSpec{3} = {'CAR_TT', 'CAR_TT x ga', 'CAR_TT x purpose', 'logt(CAR_CO)', 'logt(CAR_CO) x age', 'logt(CAR_CO) x purpose'};

% % spec8: testing removing variables not included in spec given by ARD
% fitSpec{1} = {'TRAIN_ASC', 'logt(TRAIN_TT)', 'logt(TRAIN_TT) x ga', 'logt(TRAIN_CO)', 'logt(TRAIN_CO) x ga', 'logt(TRAIN_CO) x purpose', 'logt(TRAIN_CO) x age', 'logt(TRAIN_HE)'};
% fitSpec{2} = {'SM_ASC', 'SM_ASC x ga', 'SM_ASC x age', 'logt(SM_TT)', 'logt(SM_CO)', 'logt(SM_CO) x ga', 'logt(SM_CO) x purpose'};
% fitSpec{3} = {'CAR_TT', 'CAR_TT x ga', 'CAR_TT x age', 'CAR_TT x purpose', 'logt(CAR_CO)', 'logt(CAR_CO) x age', 'logt(CAR_CO) x purpose'};

% % spec9: exact copy of spec given by ARD
% fitSpec{1} = {'logt(TRAIN_TT) x ga', 'TRAIN_ASC', 'logt(TRAIN_CO)', 'logt(TRAIN_CO)', 'logt(TRAIN_CO) x purpose', 'logt(TRAIN_CO) x ga', 'TRAIN_CO', 'logt(TRAIN_CO) x age', 'TRAIN_CO x purpose', 'logt(TRAIN_HE)'};
% fitSpec{2} = {'logt(SM_CO) x ga', 'logt(SM_CO) x purpose', 'logt(SM_CO)', 'logt(SM_TT)', 'SM_CO', 'SM_ASC x age', 'SM_CO x purpose'};
% fitSpec{3} = {'logt(CAR_CO)', 'CAR_TT x ga', 'logt(CAR_TT) x purpose', 'logt(CAR_CO) x age', 'CAR_CO x purpose', 'CAR_TT x age'};

% % spec10: exact copy of spec given by ARD - more more!
% fitSpec{1} = {'logt(TRAIN_TT) x ga', 'TRAIN_ASC', 'logt(TRAIN_CO)', 'logt(TRAIN_CO) x purpose', 'logt(TRAIN_CO) x ga', 'TRAIN_CO', 'logt(TRAIN_CO) x age', 'TRAIN_CO x purpose', 'logt(TRAIN_HE)'};
% fitSpec{2} = {'logt(SM_CO) x ga', 'logt(SM_CO) x purpose', 'logt(SM_CO)', 'logt(SM_TT)', 'SM_CO', 'SM_ASC x age', 'SM_CO x purpose', 'SM_ASC x ga'};
% fitSpec{3} = {'logt(CAR_CO)', 'CAR_TT x ga', 'logt(CAR_TT) x purpose', 'logt(CAR_CO) x age', 'CAR_CO x purpose', 'CAR_TT x age'};

% generate dataset (i.e. all variable transformation and interactions)
fprintf('\nGenerating dataset with all possible variable transformation and interactions...\n');
N = size(Y_true,1);
nChoices = max(Y_true); % number of choices/classes
% create feature transformations
D = zeros(nChoices,1);
groups = cell(nChoices,1);
X = cell(nChoices,1);
for c=1:nChoices
    groups{c} = [];
    X{c} = [];
    for d=1:length(fitSpec{c})
        varName = fitSpec{c}{d};
        %fprintf('choice %s: adding variable %s\n', alternative_names{c}, varName);
        if contains(varName, ' x ')
            [matches,~] = strsplit(varName,'\s* x \s*','DelimiterType','RegularExpression');
            var = eval(matches{1});
            inter_vars = cell(length(matches) - 1, 1);
            for i=1:length(inter_vars)
                inter_vars{i} = eval(matches{i+1});
            end
            res = [];
            for n=1:N
                if length(inter_vars) == 1
                    dim = max(inter_vars{1})-1; % last column is unnecessary
                    vec = zeros(1,dim);
                    sn = inter_vars{1}(n);
                    if sn <= dim
                        vec(sn) = var(n);
                    end
                elseif length(inter_vars) == 2
                    dim = (max(inter_vars{1})-1)*(max(inter_vars{2})-1); % last column is unnecessary
                    vec = zeros(1,dim);
                    sn1 = inter_vars{1}(n);
                    if sn1 < max(inter_vars{1})
                        sn2 = inter_vars{2}(n);
                        if sn2 < max(inter_vars{2})
                            sn = (sn1-1)*(max(inter_vars{2})-1) + sn2;
                            vec(sn) = var(n);
                        end
                    end
                else
                    error('Not implemented');
                end
                res = [res; vec];
            end
            
            D(c) = D(c) + dim;
            groups{c} = [groups{c}, d*ones(1,dim)];
            X{c} = [X{c}, res];
        else
            var = eval(varName);
            dim = size(var,2);
            D(c) = D(c) + dim;
            groups{c} = [groups{c}, d*ones(1,dim)];
            X{c} = [X{c}, var];
        end
    end
end
fprintf('Di*Kd=[%d,%d,%d]\n', D(1), D(2), D(3));
fprintf('Total variables to test: %d\n', sum(D));

% pre-compute some variables/statistics required for later
Di = zeros(nChoices,1);
Kd = cell(nChoices,1);
Dgroup = cell(nChoices,1);
for c=1:nChoices
    Di(c) = max(groups{c});
    Kd{c} = zeros(Di(c),1);
    for k=1:Di(c)
        Kd{c}(k) = sum(groups{c}==k);
    end
    Dgroup{c} = zeros(D(c),1);
    for d=1:D(c)
        Dgroup{c}(d) = Kd{c}(groups{c}(d));
    end
end
fprintf('Di=[%d,%d,%d]\n', Di(1), Di(2), Di(3));

% assign names to all generated variables
varNames = cell(nChoices,1);
for c=1:nChoices
    varNames{c} = cell(D(c),1);
    ix = 1;
    for d=1:Di(c)
        for k=1:Kd{c}(d)
            varNames{c}{ix} = sprintf('%s_%d',fitSpec{c}{d},k);
            ix = ix + 1;
        end
    end
end

% % standardize data
% for c=1:nChoices
%     meanX = mean(X{c},1);
%     stdX = std(X{c},1);
%     meanX(stdX == 0) = 0; % for bias terms
%     stdX(stdX == 0) = 1; 
%     %X{c} = (X{c} - meanX) ./ stdX;
%     %X{c}(:,1) = ones(1,N); % fix bias terms
% end

Y = Y_true;

% write dataset with all features used in the specification to disk (long format)
fileID = fopen([outdir dataName '.csv'],'w');
fprintf(fileID, 'ID,CHOICE,AV_TRAIN,AV_SM,AV_CAR');
for c=1:nChoices
    for i=1:length(varNames{c})
        fprintf(fileID, ',%s', varNames{c}{i});
    end
end
fprintf(fileID, '\n');
for n=1:N
    fprintf(fileID, '%d,%d', ids(n), Y_true(n));
    fprintf(fileID, ',%d,%d,%d', availableChoices(n,1), availableChoices(n,2), availableChoices(n,3));
    for c=1:nChoices
        for i=1:length(varNames{c})
            fprintf(fileID, ',%.3f', X{c}(n,i));
        end
    end
    fprintf(fileID, '\n');
end
fclose(fileID);

% ----------------- fit DCM model on full dataset

% run simple DCM model 
fprintf('Fitting DCM model on full dataset using MLE...\n');
Y_onehot = full(ind2vec(Y', nChoices))';
for c=1:nChoices
    theta{c} = zeros(D(c),1);
end

% fit DCM with MLE
theta_optim_full = minimize(theta, @neglog_DCM, -20000, X, Y, Y_onehot, availableChoices);
[llik_full,~] = neglog_DCM(theta_optim_full, X, Y, Y_onehot, availableChoices);
fprintf('Log-lik full dataset: %.3f\n', -llik_full);


% ----------------- fit DCM on trainset and measure performance on testset


% train/test split
Ntr = floor(0.7*N);
Xtr = cell(3,1);
Xts = cell(3,1);
for c=1:nChoices
	Xtr{c} = X{c}(1:Ntr,:);
    Xts{c} = X{c}((Ntr+1):end,:);
end
Ytr = Y(1:Ntr,:);
Ytr_onehot = full(ind2vec(Ytr', nChoices))';
Yts = Y((Ntr+1):end,:);
availableChoicesTr = availableChoices(1:Ntr,:);
availableChoicesTs = availableChoices((Ntr+1):end,:);

Ntr = size(Xtr{1},1);
Nts = size(Xts{1},1);


% run simple DCM model with all the possible variables/features
fprintf('Fitting DCM to sampled artificial choices using MLE...\n');
Y_onehot = full(ind2vec(Ytr', nChoices))';
Y_onehot_test = full(ind2vec(Yts', nChoices))';
for c=1:nChoices
    theta{c} = zeros(D(c),1);
end

% fit DCM with MLE
theta_optim_train = minimize(theta, @neglog_DCM, -20000, Xtr, Ytr, Y_onehot, availableChoicesTr);
[llik_train,~] = neglog_DCM(theta_optim_train, Xtr, Ytr, Y_onehot, availableChoicesTr);
[llik_test,~] = neglog_DCM(theta_optim_train, Xts, Yts, Y_onehot_test, availableChoicesTs);
fprintf('Log-likelihood full dataset: %.3f\n', -llik_full);
fprintf('Log-likelihood train: %.3f\n', -llik_train);
fprintf('Log-likelihood test: %.3f\n', -llik_test);

% evaluate trainset accuracy
S = DCM(theta_optim_train, Xtr, availableChoicesTr);
[~,preds_tr] = max(S,[],2);
train_acc = sum(preds_tr == Ytr) / length(Ytr);
fprintf('Train accuracy: %.3f\n', train_acc);

% evaluate testset accuracy
S = DCM(theta_optim_train, Xts, availableChoicesTs);
[~,preds_ts] = max(S,[],2);
test_acc = sum(preds_ts == Yts) / length(Yts);
fprintf('Test accuracy: %.3f\n', test_acc);


fileID = fopen([outdir dataName '_accuracy.txt'],'w');
fprintf(fileID, 'Log-likelihood full dataset: %.3f\n', -llik_full);
fprintf(fileID, 'Log-likelihood train: %.3f\n', -llik_train);
fprintf(fileID, 'Log-likelihood test: %.3f\n', -llik_test);
fprintf(fileID, 'Train accuracy: %.3f\n', train_acc);
fprintf(fileID, 'Test accuracy: %.3f\n', test_acc);
fclose(fileID);


function ret = logt(vec)
    ret = log(vec+1);
end

function ret = boxt(vec)
    ret = boxcox(vec+1);
end

function ret = segt(vec,k)
    I = eye(k);
    ret = I(kmeans(vec, k),:);
    ret = ret(:,1:(k-1)); % don't add last column - it should be captured by the bias term
end

