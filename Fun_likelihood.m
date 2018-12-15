function likelihood = Fun_likelihood(eta,h,theta,tau,K,q_0,q_1,alpha,beta) 
  
        probit_F=zeros(K,1);
        PDF = @(x) alpha*beta/2/gamma(1/beta)*exp(-(alpha*abs(x)).^beta);           % function f(x)
        CDF = @(x) integral(PDF,-inf,x);                             % function F(x)
        
        for i=1:K
            probit_F(i)=CDF(h(i)*theta-tau(i));
        end
        
       likelihood = eta'*log(q_0+(1-q_0-q_1)*probit_F)+(1-eta)'*log(1-q_0-(1-q_0-q_1)*probit_F);
end