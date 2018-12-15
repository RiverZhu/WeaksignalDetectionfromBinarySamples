clc;
clear;
close all;
%% Basic parameters
MC = 2000;                                                         % Monte Carle times
N=2000;     
alpha=1;                                                              % scaling parameter 
beta_sam=[1.5 2 4 8];
PFA = [0.001;0.05;0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9;1]; %Pfa SAMPLES
eta00=zeros(1,MC);
eta10=zeros(1,MC);
eta0s=zeros(1,MC);
eta1s=zeros(1,MC);
Pd0G = zeros(size(PFA));
Pd0R = zeros(size(PFA));
Pd0t = zeros(size(PFA));
PdsG = zeros(size(PFA));
PdsR = zeros(size(PFA));
Pdst = zeros(size(PFA));
w0=zeros(N,MC);
PD0G=zeros(length(PFA),length(beta_sam));
PDSG=zeros(length(PFA),length(beta_sam));
PD0R=zeros(length(PFA),length(beta_sam));
PDSR=zeros(length(PFA),length(beta_sam));
PD0T=zeros(length(PFA),length(beta_sam));
PDST=zeros(length(PFA),length(beta_sam));
step=200;
q_0=0.7;                                                               % flipping probabilities
q_1=0;
lambda_defult=1/2*pi/(1-q_0-q_1)^2;                   % default FI, Chapter 6, Kay                                                          
theta = sqrt(lambda_defult/(2*alpha^2*N)); 
h = 1;

for beta_index=1:length(beta_sam)
    beta=beta_sam(beta_index);
    %% Generate GGN
for i=1:N/step
    for j=1:MC/step
        waitbar((100*(beta_index-1)+10*(i-1)+j)/400);
        w0(1+(i-1)*step : i*step,1+(j-1)*step : j*step) = GGN_step(alpha,beta,step,step);                         % GGN
    end
end
w1=w0+h*theta;
%% Calculate tau*
mct=100;                                                             % max circulation times when calculating step length 'ct'
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
ct=1;                                                                   % circulation time

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

%% Parameter tests
tau0 = 0;
taus= -optimum;

count00=sum(w0>tau0); % Quantization
count10=sum(w1>tau0);
count0s=sum(w0>taus);
count1s=sum(w1>taus);

for mc=1:MC % flipping
    eta00(mc)=( sum( rand(1,count00(mc))>q_1*ones(1,count00(mc)))+sum (rand(1,N-count00(mc))<q_0*ones(1,N-count00(mc)))  )/N;        
    eta10(mc)=( sum( rand(1,count10(mc))>q_1*ones(1,count10(mc)))+sum( rand(1,N-count10(mc))<q_0*ones(1,N-count10(mc)))  )/N;         
    eta0s(mc)=( sum( rand(1,count0s(mc))>q_1*ones(1,count0s(mc)))+sum( rand(1,N-count0s(mc))<q_0*ones(1,N-count0s(mc)))  )/N;        
    eta1s(mc)=( sum( rand(1,count1s(mc))>q_1*ones(1,count1s(mc)))+sum( rand(1,N-count1s(mc))<q_0*ones(1,N-count1s(mc)))  )/N;     
end

% GLRT, under H0/H1 and tau0/tau*
T00 = N*( eta00.*log(eta00)+(1-eta00).*log(1-eta00)-(1-eta00).*log(1-q_0-(1-q_0-q_1)*CDF(-tau0))-eta00.*log(q_0+(1-q_0-q_1)*CDF(-tau0))); 
T0s = N*( eta0s.*log(eta0s)+(1-eta0s).*log(1-eta0s)-(1-eta0s).*log(1-q_0-(1-q_0-q_1)*CDF(-taus))-eta0s.*log(q_0+(1-q_0-q_1)*CDF(-taus))); 
T10 = N*( eta10.*log(eta10)+(1-eta10).*log(1-eta10)-(1-eta10).*log(1-q_0-(1-q_0-q_1)*CDF(-tau0))-eta10.*log(q_0+(1-q_0-q_1)*CDF(-tau0))); 
T1s = N*( eta1s.*log(eta1s)+(1-eta1s).*log(1-eta1s)-(1-eta1s).*log(1-q_0-(1-q_0-q_1)*CDF(-taus))-eta1s.*log(q_0+(1-q_0-q_1)*CDF(-taus))); 
% FI
temp0 = q_0+(1-q_0-q_1)*CDF(-tau0); % F(-tau)
FI0 =N*h^2*(1-q_0-q_1)^2*h^2*PDF(-tau0)^2/temp0/(1-temp0); % Fisher Information
temps = q_0+(1-q_0-q_1)*CDF(-taus);
FIs = N*h^2*(1-q_0-q_1)^2*h^2*PDF(-taus)^2/temps/(1-temps);
% Rao test
der00=N*h*(1-q_0-q_1)*PDF(0)*(eta00/temp0 - (1-eta00)/(1-temp0));
R00=der00.^2/FI0;
der10=N*h*(1-q_0-q_1)*PDF(0)*(eta10/temp0 - (1-eta10)/(1-temp0));
R10=der10.^2/FI0;

der0s=N*h*(1-q_0-q_1)*PDF(0)*(eta0s/temps - (1-eta0s)/(1-temps));
R0s=der0s.^2/FIs;
der1s=N*h*(1-q_0-q_1)*PDF(0)*(eta1s/temps - (1-eta1s)/(1-temps));
R1s=der1s.^2/FIs;

% ROC
for i = 1:length(PFA) 
    P_fa=PFA(i); 
    
    T00_sort = sort(T00,'descend');
    T0s_sort = sort(T0s,'descend');
    g0g = T00_sort(floor(MC*P_fa));        % gamma, GLRT
    gsg = T0s_sort(floor(MC*P_fa));  
    Pd0G(i) = sum(T10>g0g)/MC;            
    PdsG(i) = sum(T1s>gsg)/MC;     
    
    R00_sort = sort(R00,'descend');
    R0s_sort = sort(R0s,'descend');
    g0r = R00_sort(floor(MC*P_fa));        % gamma, Rao test
    gsr = R0s_sort(floor(MC*P_fa));  
    Pd0R(i) = sum(R10>g0r)/MC;            
    PdsR(i) = sum(R1s>gsr)/MC;       
    
    X_fa=chi2inv(1-P_fa,1);  % gamma, asymptotic performance
    Pd0t(i) = ncx2cdf(X_fa,1, theta^2*FI0,'upper'); %Theoretical results
    Pdst(i) = ncx2cdf(X_fa,1, theta^2*FIs,'upper');
end
    PD0G(1:length(PFA),beta_index)= Pd0G;
    PDSG(1:length(PFA),beta_index)= PdsG;
    PD0R(1:length(PFA),beta_index)= Pd0R;
    PDSR(1:length(PFA),beta_index)= PdsR;
    PD0T(1:length(PFA),beta_index)=Pd0t;
    PDST(1:length(PFA),beta_index)=Pdst;
end
%% figure
alw = 0.75;    % AxesLineWidth
fsz = 10;      % Fontsize
lw = 1.0;      % LineWidth
msz = 10;       % MarkerSize
% figure(1)
% plot(P_fa_samples,P_D,'-bo',P_fa_samples,P_Ds,'-ks',P_fa_samples,P_d_theory,'-.rx',P_fa_samples,P_d_theorys,'--m+','LineWidth',lw,'MarkerSize',msz)
% xlabel('P_{fa}','Fontsize',fsz)
% ylabel('P_D','Fontsize',fsz)
% % text(0.5*max(K_sam)+0.5*min(K_sam),0.85*min(min(F1_sub1))+0.15,'(a) \beta=1.5','Fontsize',15);
% legend('GLRT, \tau=0','GLRT, \tau=\tau^*','Theoretical, \tau=0','Theoretical, \tau=\tau^*');
% grid on;
close all;
figure(1)
subplot(2,2,1)
% plot(PFA,PD0G(:,1),'-bo',PFA,PDSG(:,1),'-bs',PFA,PD0R(:,1),'-r+',PFA,PDSR(:,1),'-rx',PFA,PD0T(:,1),'--k',PFA,PDST(:,1),'-.k','LineWidth',lw,'MarkerSize',msz)
plot(PFA,PD0G(:,1),'-bo',PFA,PDSG(:,1),'-ks',PFA,PD0T(:,1),'-.rx',PFA,PDST(:,1),'--m+','LineWidth',lw,'MarkerSize',msz)
xlabel('P_{FA}','Fontsize',fsz)
ylabel('P_D','Fontsize',fsz)
text(0.75,0.1,'(a) \beta=1.5','Fontsize',15);
% legend('GLRT, \tau=0','GLRT, \tau=\tau^*','Rao test, \tau=0','Rao test, \tau=\tau^*','Theoretical, \tau=0','Theoretical, \tau=\tau^*');
legend('GLRT, \tau=0','GLRT, \tau=\tau^*','Theoretical, \tau=0','Theoretical, \tau=\tau^*');
grid on;
subplot(2,2,2)
plot(PFA,PD0G(:,2),'-bo',PFA,PDSG(:,2),'-ks',PFA,PD0T(:,2),'-.rx',PFA,PDST(:,2),'--m+','LineWidth',lw,'MarkerSize',msz)
xlabel('P_{FA}','Fontsize',fsz)
ylabel('P_D','Fontsize',fsz)
text(0.75,0.1,'(b) \beta=2','Fontsize',15);
grid on;
subplot(2,2,3)
plot(PFA,PD0G(:,3),'-bo',PFA,PDSG(:,3),'-ks',PFA,PD0T(:,3),'-.rx',PFA,PDST(:,3),'--m+','LineWidth',lw,'MarkerSize',msz)
xlabel('P_{FA}','Fontsize',fsz)
ylabel('P_D','Fontsize',fsz)
text(0.75,0.1,'(c) \beta=4','Fontsize',15);
grid on;
subplot(2,2,4)
plot(PFA,PD0G(:,4),'-bo',PFA,PDSG(:,4),'-ks',PFA,PD0T(:,4),'-.rx',PFA,PDST(:,4),'--m+','LineWidth',lw,'MarkerSize',msz)
xlabel('P_{FA}','Fontsize',fsz)
ylabel('P_D','Fontsize',fsz)
text(0.75,0.1,'(d) \beta=8','Fontsize',15);
grid on;

% save ROC_4beta_20180427;




