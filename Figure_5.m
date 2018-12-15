clc;
clear;
close all;
%% Parallel
% parpool('open','local',8);
%% Initialization (threshold design)
alpha=1;                                                              % scaling parameter 
beta=2.779;
q_0=0.3;                                                               % flipping probabilities
q_1=0;
%% Calculate tau*
mct=20;                                                             % max circulation times when calculating step length 'ct'
X=zeros(1,mct);                                                    %  t is the variables
G=zeros(1,mct);                                                   % value of function G(x)
G_D=zeros(1,mct);                                               % value of function G'(x)
c_1=alpha*beta/2/gamma(1/beta);                       % replacement for simplicity
c_2=(2*q_0-1)/2/(1-q_0-q_1);
c_3=1/4/(1-q_0-q_1)^2;
PDF = @(x) c_1*exp(-(alpha*abs(x)).^beta);          % function f(x)
CDF = @(x) integral(PDF,-inf,x);                           % function F(x)
FUN_G=@(x) -c_1^2*exp(-2*(alpha*abs(x)).^beta)./(1/4-((1-q_0-q_1)*integral(PDF,-inf,x)-1/2+q_0).^2);   % function G(x)
FUN_G_D=@(x) -2*c_1^2*exp(-2*(alpha*abs(x)).^beta)./(1/4-((1-q_0-q_1)*integral(PDF,-inf,x)-1/2+q_0).^2).* ... % the first order derivative of G(x)
( -alpha^beta*beta*abs(x).^(beta-1) + c_1*exp(-(alpha*abs(x)).^beta).*(integral(PDF,-inf,x)+c_2) ./ (c_3-(integral(PDF,-inf,x)+c_2).^2) ); 
X(1)=0.01/alpha;                                                  % initial point
stop_criterion=(1e-5)*alpha^3;                            % stop criterion
lambda=stop_criterion+1;              
ct=1;                                                                  % circulation time
while(ct<=mct&&lambda>stop_criterion)% Gradient descent method
    G_D(ct)=FUN_G_D(X(ct));
    t=1/alpha^4;
    while(X(ct)+t*(-G_D(ct))<0)
        t=0.5*t;
    end;
    while(  FUN_G(X(ct)+t*(-G_D(ct)))  >  (FUN_G(X(ct))  +  0.2*t*(-G_D(ct)^2))  )
        t=0.4*t;
    end
    X(ct+1)=X(ct)-t*G_D(ct);
    lambda=abs(G_D(ct));
    ct=ct+1;
end
optimum=X(ct-1); % optimal threshold
%% Initialization (detection)
options = optimset('Largescale','off','GradObj','off','Hessian','off','MaxFunEvals',2000,'MaxIter',1000,'Display','off','DerivativeCheck','on'); 
MC = 1000;                                                         % Monte Carle times
N=50;  
K=50;
P_fa_samples = [0.002;0.05;0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9;1]; %Pfa                                               
theta=0.05;
T00=zeros(1,MC); % GLRT, T= l_theta-l_0
T10=zeros(1,MC);
T0s=zeros(1,MC);
T1s=zeros(1,MC);
R00=zeros(1,MC); % Rao test
R10=zeros(1,MC);
R0s=zeros(1,MC);
R1s=zeros(1,MC);
Pd0g = zeros(size(P_fa_samples));  %GLRT results of P_d under tau=0
Pdsg = zeros(size(P_fa_samples));
Pd0r = zeros(size(P_fa_samples));  %GLRT results of P_d under tau=0
Pdsr = zeros(size(P_fa_samples));
Pd0t = zeros(size(P_fa_samples));   % theoretical results
Pdst = zeros(size(P_fa_samples));
%% Generate h and tau
h=zeros(N,K); %
for i=1:N
    for j=1:K
        h(i,j)=sin(1.676*i-2.514*j);
    end
end
tau0 = zeros(N,K);  % tau =0 or optimum
taus= -optimum*ones(N,K);
%% Calculate FI
temp0 = q_0+(1-q_0-q_1)*CDF(0);                        % F(-tau)
temps = q_0+(1-q_0-q_1)*CDF(optimum);
FI0 = (1-q_0-q_1)^2*PDF(0)^2/temp0/(1-temp0)*sum(sum(h.*h)); % Fisher Information
FIs = (1-q_0-q_1)^2*PDF(optimum)^2/temps/(1-temps)*sum(sum(h.*h));
%% Monte Carlo trials
% parfor mc=1:MC
parfor mc=1:MC
    waitbar(mc/MC);
    w0 =zeros(N,K);    % noise
    for i=1:N
        for j=1:K
            w0(i,j) = GGN(alpha,beta);   % generate GGN
        end
    end
    w1=w0+h*theta; % noisy samples
    % quantization
    qs00=(w0>tau0);   % quantized samples in zero signal zero threshold = b_ij
    qs10=(w1>tau0);
    qs0s=(w0>taus);
    qs1s=(w1>taus);
    % flipping
    fs00 =qs00.*(rand(N,K)>q_1*ones(N,K))+(1-qs00).*(rand(N,K)<q_0*ones(N,K)); % flipped samples = u_ij
    fs10 =qs10.*(rand(N,K)>q_1*ones(N,K))+(1-qs10).*(rand(N,K)<q_0*ones(N,K));         
    fs0s =qs0s.*(rand(N,K) >q_1*ones(N,K))+(1-qs0s).*(rand(N,K)<q_0*ones(N,K));        
    fs1s =qs1s.*(rand(N,K) >q_1*ones(N,K))+(1-qs1s).*(rand(N,K)<q_0*ones(N,K));  
    %% GLRT, tau = 0
    theta00 = fminunc(@(theta)Fun_est(fs00,h,theta,tau0,N,K,q_0,q_1,alpha,beta),0,options);
    theta10 = fminunc(@(theta)Fun_est(fs10,h,theta,tau0,N,K,q_0,q_1,alpha,beta),0,options);
    T00(mc)= Fun_llh(fs00,h,theta00,tau0,N,K,q_0,q_1,alpha,beta)-Fun_llh(fs00,h,0,tau0,N,K,q_0,q_1,alpha,beta);
    T10(mc)= Fun_llh(fs10,h,theta10,tau0,N,K,q_0,q_1,alpha,beta)-Fun_llh(fs10,h,0,tau0,N,K,q_0,q_1,alpha,beta);    
    %% GLRT, tau = tau*
    theta0s = fminunc(@(theta)Fun_est(fs0s,h,theta,taus,N,K,q_0,q_1,alpha,beta),0,options);
    theta1s = fminunc(@(theta)Fun_est(fs1s,h,theta,taus,N,K,q_0,q_1,alpha,beta),0,options);
    T0s(mc)= Fun_llh(fs0s,h,theta0s,taus,N,K,q_0,q_1,alpha,beta)-Fun_llh(fs0s,h,0,taus,N,K,q_0,q_1,alpha,beta);
    T1s(mc)= Fun_llh(fs1s,h,theta1s,taus,N,K,q_0,q_1,alpha,beta)-Fun_llh(fs1s,h,0,taus,N,K,q_0,q_1,alpha,beta);
    %% Rao test, tau = 0
    der00=(1-q_0-q_1)*PDF(0)*sum(sum(h.*( fs00/temp0 - (1-fs00)/(1-temp0) )));
    R00(mc)=der00^2/FI0;
    der10=(1-q_0-q_1)*PDF(0)*sum(sum(h.*( fs10/temp0 - (1-fs10)/(1-temp0) )));
    R10(mc)=der10^2/FI0;
    %% Rao test, tau = tau*
    der0s=(1-q_0-q_1)*PDF(0)*sum(sum(h.*( fs0s/temps - (1-fs0s)/(1-temps) )));
    R0s(mc)=der0s^2/FIs;
    der1s=(1-q_0-q_1)*PDF(0)*sum(sum(h.*( fs1s/temps - (1-fs1s)/(1-temps) )));
    R1s(mc)=der1s^2/FIs;
end
%% Pd - Pfa   
T00_sort=sort(T00,'descend');                               % sort T
T0s_sort=sort(T0s,'descend');
R00_sort=sort(R00,'descend');                               % sort R
R0s_sort=sort(R0s,'descend');

for i = 1:length(P_fa_samples) 
    P_fa=P_fa_samples(i); 
    g0g = T00_sort(floor(MC*P_fa));                  %  GLRT, gamma about P_d P_fa under tau=0
    gsg = T0s_sort(floor(MC*P_fa));      
    g0r = R00_sort(floor(MC*P_fa));                  %  Rao test, gamma 
    gsr = R0s_sort(floor(MC*P_fa));        
    Pd0g(i) = sum(T10>g0g)/MC;                     % GLRT
    Pdsg(i) = sum(T1s>gsg)/MC;     
    Pd0r(i) = sum(R10>g0r)/MC;                       % Rao test
    Pdsr(i) = sum(R1s>gsr)/MC;     
    X_fa=chi2inv(1-P_fa,1);
    Pd0t(i) = ncx2cdf(X_fa,1,theta^2*FI0,'upper');                 % theoretical results, chi-square
    Pdst(i) = ncx2cdf(X_fa,1,theta^2*FIs,'upper');
end
%% figure
close all;
alw = 0.75;    % AxesLineWidth
fsz = 15;      % Fontsize
lw = 1;      % LineWidth
msz =10;       % MarkerSize
figure(1)
close all;
plot(P_fa_samples,Pd0g,'-bo',P_fa_samples,Pdsg,'-bs',P_fa_samples,Pd0r,'-r+',P_fa_samples,Pdsr,'-rx','LineWidth',lw,'MarkerSize',msz)
hold on;
plot(P_fa_samples,Pd0t,'--k',P_fa_samples,Pdst,':k','LineWidth',2)
xlabel('P_{FA}','Fontsize',fsz)
ylabel('P_D','Fontsize',fsz)
grid on;
leg=legend('GLRT, \tau=0','GLRT, \tau=\tau^*','Rao test, \tau=0','Rao test, \tau=\tau^*','Theoretical, \tau=0','Theoretical, \tau=\tau^*');
set(leg,'Fontsize',12);
% save wavevector_20180427;




