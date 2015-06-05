%--------------------------------------------------------------------------
% This is for the CIVL4140&7140 modelling project (2015)
% Matlab script for computing one-dimensional river flows subject to
% increased discharge from upsteram over a rainfall event.
% A rectangular cross-section of the channel has been assumed.
% Governing equations: St Venant equations
% Numerical scheme: Lax-Friedrichs
% Notice:
% 1. Please do not change the name of the variables so that we can assit you 
% more efficiently in building model 2 and 3. 
% 2. Use compare function in Matlab m file editor to heilight all the changes
% made during  model 2 and 3 development. This can help to identify 
% unexpected changes
%--------------------------------------------------------------------------
clear all
% input data: channel configuration, river section parameters and other
% physical parameters
n=0.03;s0=0.0002;B=10;g=9.81;L=4000;
% calculate the steady state uniform flow
h0=5;A0=h0*B;HR0=A0/(2*h0+B);
Q0=sqrt(s0*A0^2*HR0^(4/3)/n^2);
% check the flow condition: u0-c0<0 --> subcritical flow
u0=Q0/A0;
c0=sqrt(g*h0);
if(u0-c0<0)
    display('initial flow condtion: subcritical');
end
% Parameter values for describing the hydrograph at the upstream boundary
% (as a pulse-like signal)

f1=4;f2=360;f3=100;f4=2;
ucm=(1+f1)*(u0+c0); % estimate of the maximum advancing
                     % characteristic speed for the determination of a
                     % proper time step to ensure a stable solution
% spatial grid size and time step
dx=10;
dt=0.8*dx/ucm;
lm=dt/dx;

% discretise the river section
x=0:dx:L+dx;
nx=length(x);
% set the initial condition: U1 is the first element in the U vector
U1=zeros(1,nx)+A0;
U2=zeros(1,nx)+Q0;
np=0;

for i=2:10^7
    t=(i-1)*dt;
    F1=U2(1,:);
    F2=U2(1,:).^2./U1(1,:)+g*U1(1,:).^2/(2*B); % add . to any operator will 
                                                    % make the operation
                                                    % done on an element by
                                                    % element basis.
    R1=zeros(1,nx);
    HR=U1(1,:)./(2*U1(1,:)/B+B);
    R2=g*U1(1,:).*(s0-n^2*U2(1,:).^2./(U1(1,:).*HR.^(4/3)));
    for j=2:nx-1
        U1(2,j)=0.5*(U1(1,j-1)+U1(1,j+1))-0.5*lm*(F1(j+1)-F1(j-1))+dt*R1(j);
        U2(2,j)=0.5*(U2(1,j-1)+U2(1,j+1))-0.5*lm*(F2(j+1)-F2(j-1))+dt*R2(j);
    end
    U2(2,1)=Q0*(1+f1*exp(-(abs(t-f2)/f3)^f4)); % set the U2 value for the upstream
                                                     % boundary node
                                                     % according the
                                                     % specified discharge
% calculate the U1 (A) value for the upstream boundary using the
% characteristic method. An interation has been introduced to determine xr.
    ur=U2(1,2)/U1(1,2);
    cr=sqrt(g*U1(1,2)/B);
    dxr=10;
    while dxr>dx/1000
        xr=x(1)-(ur-cr)*dt;
        QR=U2(1,2)-(U2(1,2)-U2(1,1))/dx*(x(2)-xr);
        AR=U1(1,2)-(U1(1,2)-U1(1,1))/dx*(x(2)-xr);
        ur=QR/AR;
        cr=sqrt(g*AR/B);
        xrm=x(1)-(ur-cr)*dt;
        dxr=abs(xr-xrm);
    end


    xr      = 0.5*(xr+xrm);
    QR      = U2(1,2)-(U2(1,2)-U2(1,1))/dx*(x(2)-xr);
    AR      = U1(1,2)-(U1(1,2)-U1(1,1))/dx*(x(2)-xr);
    wp      = 2*AR/B+B;
    RR      = g*AR*(s0-n^2*QR^2/(AR^2*(AR/wp)^(4/3)));
    ur      = QR/AR;
    cr      = sqrt(g*AR/B);
    U1(2,1) = ((U2(2,1)-QR)-dt*RR)/(ur+cr)+AR;

% the downstream boundary condition: transmissive boundary is assumed
    U1(2,nx)=U1(2,nx-1);
    U2(2,nx)=U2(2,nx-1);
% result visualisation
    if(rem(i,100)==0)
        np=np+1;
        tp(np)=t;
        U1p(np,:)=U1(2,:);
        U2p(np,:)=U2(2,:);
        subplot(2,2,1),plot(tp,U1p(:,[1 10 50])/B),title('h at x = 0, 100 and 500 m')
        axis([0 max(t) h0-0.5 h0+2])
        subplot(2,2,2),plot(tp,U2p(:,[1 10 50])./U1p(:,[1 10 50])), title('u at x = 0, 100 and 500 m')
        axis([0 max(t) u0-0.2 u0+f1*Q0/A0])
        subplot(2,1,2),plot(x,U1p(np,:)/B),
        axis([0 L h0-1 h0+2])
        pause(1)
    end
    U1(1,:)=U1(2,:);
    U2(1,:)=U2(2,:);
end
    
