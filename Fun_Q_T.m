function [neg_log_like,gradient_theta] = Fun_Q_T(Eta,h,theta,tau,K,N,q0,q1,alpha,beta) 
        
        probit_F=zeros(K,1);
        probit_f=zeros(K,1);

        PDF = @(x) alpha*beta/2/gamma(1/beta)*exp(-(alpha*abs(x)).^beta);           % function f(x)
        CDF = @(x) integral(PDF,-inf,x);                             % function F(x)
        
        for i=1:K
            probit_F(i)=CDF(h(i)*theta-tau(i));
            probit_f(i)=PDF(h(i)*theta-tau(i));
        end

    neg_log_like = -N*((Eta'* log( q0+(1-q0-q1)*probit_F) + (1-Eta)'* log(1-q0-(1-q0-q1)*probit_F)));
    
    gradient_theta = -N*(1-q0-q1)*( Eta./(q0+(1-q0-q1)*probit_F)-(1-Eta)./(1-q0-(1-q0-q1)*probit_F) )'*probit_f; % derivative based on -loglike
    
end