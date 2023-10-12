function [accuracy]=DPCLS(train_data,train_p_target,test_data,test_target,optmparameter)

lambda=optmparameter.lambda;
alpha=optmparameter.alpha;
beta=optmparameter.beta;
k=optmparameter.k;
par=mean(pdist(train_data));
ker='rbf';

F=train_p_target./(sum(train_p_target,2));
S=Construct_similarity_matrix(train_data,k);
D0=Construct_D0(train_p_target);

n=size(S,1);
tol=1e-8;
max_iter=800;
rho=1.1;
mu=1e-4;
max_mu=1e10;

S=(S+S')/2;
D_s=sum(S,2);
L=diag(D_s)-S;
A=zeros(n);
Phi1=zeros(n);

for iter=1:max_iter
    % W subproblem
    [train_output,test_output]=kernelRidgeRegression(train_data,F,test_data,lambda,par,ker);

    % F subproblem
    %%% small data set
    F=Update_F(train_output,train_p_target,A,alpha);
    %%% large data set
    %F=Update_F_largedataset(F,train_output,train_p_target,A,alpha);

    % D subproblem
    D=(mu*A-Phi1)/(2*beta*L+mu*eye(n));

    % A subproblem
    A=D+(Phi1/mu)-(alpha/mu).*F*F';
    A(D0==1)=1;
    A(A<0)=0;
    A(A>1)=1;

    d=D-A;

    chg=max(max(abs(d)));

    if chg<tol
        break;
    end

    if mod(iter,50)==0
        disp([num2str(iter),' Err D-A: ',num2str(norm(d,'fro'))])
    end

    Phi1=Phi1+mu*d;
    mu=min(rho*mu,max_mu);

end

accuracy=CalAccuracy(test_output,test_target);
end