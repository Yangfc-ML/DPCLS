function D=Construct_D0(train_p_target)

[m, ~]=size(train_p_target);
label_sim=1-pdist2(train_p_target,train_p_target,'cosine');
D=zeros(m,m);
D(label_sim==0)=1;

end