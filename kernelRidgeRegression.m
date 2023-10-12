function [train_outputs,test_outputs]=kernelRidgeRegression(train_data,train_target,test_data,lambda,par,ker)

[m,~]=size(train_data);
[t,~]=size(test_data);

K=kernelmatrix(ker,train_data',train_data',par);
Kt=kernelmatrix(ker,test_data',train_data',par); 
m1=ones(m,1);

U=train_target;

A=(K+lambda*eye(m,m)-1/m*m1*m1'*K)\(U-1/m*m1*m1'*U);
b=1/m*(U'-A'*K')*m1;

train_outputs=K*A+repmat(b',m,1);
test_outputs=Kt*A+repmat(b',t,1);

end