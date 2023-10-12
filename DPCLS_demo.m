load('lost sample.mat');

optmparameter.lambda=0.05;
optmparameter.alpha=1e-2;
optmparameter.beta=1e-3;
optmparameter.k=10;
list=[];

parfor i =1:10

    train_data=data(tr_idx{i},:);
    test_data = data(te_idx{i},:);
    train_p_target=partial_target(:,tr_idx{i})';
    test_target=target(:,te_idx{i})';

    [accuracy1]=DPCLS(train_data,train_p_target,test_data,test_target,optmparameter);
    list=[list,accuracy1];

end

fprintf('classification accuracy: %f std: %f\n',mean(list),std(list));



