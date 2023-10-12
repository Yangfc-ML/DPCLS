function F=Update_F_largedataset(F,train_outputs,train_p_target,D,alpha)

[p,q]=size(train_p_target);
options = optimoptions('quadprog','Display','off','Algorithm','interior-point-convex' );
lb=sparse(q,1);
Aeq=ones(1,q);
b=1;

D=(D+D')/2;
for i=1:p
    M=(2*D(i,i)+(2/alpha))*eye(q);
    t=D(i,:)'.*F;
    t_sum=sum(t)-t(i,:);
    ub=train_p_target(i,:)';
    f=quadprog(M,t_sum-2*(1/alpha)*train_outputs(i,:),[],[],Aeq,b,lb,ub,[],options);
    F(i,:)=f;
end

end