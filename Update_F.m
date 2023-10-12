function F=Update_F(train_outputs,train_p_target,D,alpha)

[p,q]=size(train_p_target);
options = optimoptions('quadprog','Display','off','Algorithm','interior-point-convex');
D=(D+D')/2;
D=2*D+(2/alpha)*eye(p);

D1=repmat({D},1,q);
M=spblkdiag(D1{:});

lb=sparse(p*q,1);
ub=reshape(train_p_target,p*q,1);
II=sparse(eye(p));
Aeq=repmat(II,1,q);
b=ones(p,1);
f=reshape(train_outputs,p*q,1);
F=quadprog(M,-2*(1/alpha)*f,[],[],Aeq,b,lb,ub,[],options);
F=reshape(F,p,q);

end