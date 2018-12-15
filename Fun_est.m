function [llh,gradient] = Fun_est(u,h,theta,tau,N,K,q0,q1,alpha,beta) 
    PDF = @(x)alpha*beta/2/gamma(1/beta)*exp(-(alpha*abs(x)).^beta);          % function f(x)
    CDF = @(x) integral(PDF,-inf,x);                           % function F(x)
    p_f=zeros(N,K);
    p_F=zeros(N,K);
    for i=1:N
        for j=1:K
            p_f(i,j) = h(i,j)*PDF(h(i,j)*theta-tau(i,j));
            p_F(i,j) = CDF(h(i,j)*theta-tau(i,j));
        end
    end
    llh = -u.* log(q0+(1-q0-q1)*p_F) - (1-u).* log(1-q0-(1-q0-q1)*p_F); 
    llh=sum(sum(llh));
    gradient = (1-q0-q1)*p_f.*(-u./(q0+(1-q0-q1)*p_F)+(1-u)./(1-q0-(1-q0-q1)*p_F)); % derivative based on -loglike
    gradient=sum(sum(gradient));
end