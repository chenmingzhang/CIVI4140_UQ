% This is the list of commands and scripts used for the MATLAB crash course

% 1. why MATLAB, why not excel? multiple dimensions, large datasets, libraries
% 2. what is MATLAB
% 3. Interface
%   3.1 current folder, workspace, command window, editor
%   3.2 check current working directory, ensure that working directory is at a local path to maximise the calculation speed.
%   3.3 use help on the daily basis.


%% below are the list of commands to run through
pwd  % show current working directory address
ls   % show current directory
help pwd % get help, second way to help is
3-5
cos(2*3.14)
clear    % clear the variable defined
clc
a=3-5
b=cos(2*pi)
a-b
c=a-b
a=a-b
d-b
a-b;   # supress the output, but the command is executed.
d=.6

b-d*c/b*a^b*a-e    % BEDMAS


size(a)
c(1,1)=1;
c(1,2)=2;
c(1,3)=3;
c(1,4)=4;
c(1,2)=c(1,1)*c(2,2);


a
size (a)
c=[2 4 6 8];
d=[1;2;3;4];
c(1,3)
d(4,1)
size(e)
size(c)
size(d)
c*a
a*d
d*c
c*d
d'*d
d.*d
d./e
d/e
d\e
e=[1 4 7 8;
9 3 1 6;
4 6 8 5;
1 7 5 3]
size(e)
e(3,4)
e(0); % index starts from 1
e(8)
e(end)
e(1,1:4)
e(1:4,1)
e(2:3,1:3)
e(:,1)
e(1:2:4,1)
e(e>2)
e(e==2)=11
e(end,end)
e*a 
e*c 
e'

% three dimensional data
p(:,:,1)=[1,2;3,4];
p(:,:,2)=[5,6;7,8];
p(:,2,2)
p(1,1,:)
squeeze(p(1,1,:));

%% end of three dimensional data

%% ----debug the following program---
% copy this section into example0.m file
clear all
e=[1 4 7 8;
9 3 1 6;
4 6 8 5;
1 7 5 3];
d=[1;2;3;4];
c=[2 4 6 8];
a=2;
ee=8;
hg=9;
h=d*e;
f=e*d;



save('a.mat')
clear
d
e
load('a.mat')
d
e

% use struct to load a workspace
st=load('a.mat')
st2=load('a.mat')

st.e
st.d

% the logical expression is important for if conditions

%% ---end debugging the program-------


%% ------- understanding if statements -------
% copy this section into example.m file
% note that only alphabetic, numbers and underscores
%  are allowed to name the m file. no spaces and minus(scores)  
%  if conditions
%     scripts
%  end
clear;
index1=-1;
 
if (index1>0 && index1<3)
  a=2;
  b=sqrt(a);
    if a>5
       d=7;
    end
elseif index1==0
  a=4;
  b=a^2;
elseif (index1<-8 || index1>=3)
  a=10;
else    
  a=3;
end


%% ---- end of understanding if statements ----

%if and logical conditions could only be used for scalars not for vectors
e=[1 4 7 8;
9 3 1 6;
4 6 8 5;
1 7 5 3];
d=[1;2;3;4];
c=[2 4 6 8];

e( e>5  &&  e <=7)=6.5

e( e<1.2 || e>8.5)=11.2


%% --------understanding loops -------------------
clear
% loops
for i=1:2:10
   j=2*i;     % no semicolon for displaying purpose
   x=sprintf('j value is %d, i value is %d.',j,i);
   disp(x)
end


% nested loops 
% use print here, and index here
y=0;
k=0;
for i=1:10
 for j=1:5
     k=k+i-j;
     y=y+1;
     x=sprintf('i is %d, j is %d, total loop number is %d.',i,j,y);
     disp(x) ;
 end
end
%% -----------end of understanding loops----------



% ----- understanding stack -----


% ----- three different method doing the same function ---
% the first
clear
c=0;
tic % tic and toc can record the excution time between the lines
for i=1:100
c=c+i;
end
toc

% the second
clear
b=0;
c=0;
tic
while b<101
   c=c+b; 
   b=b+1;
end
toc
%the third
clear
tic
d=sum(1:100);
toc


%% plot session
x=1:0.1:12;
y1=cos(x);
y2=sin(x);
y3=sin(x-1);
fig=figure;
plot(x,y1,'r-x','linewidth',2,'displayname','cos(x)');hold on;
plot(x,y2,'b:v','linewidth',3,'displayname','sin(x)');hold on;
plot(x,y3,'g-.o','linewidth',1,'displayname','sin(x-1)');hold on;
xlabel('x (m)','FontSize',10,'FontWeight','bold')
ylabel('z (m)','FontSize',12,'FontWeight','bold')
title('the result for trigonometric functions');
legend('show','Location','Southeast')
print(fig,'head.png','-dpng')   % png figure output
print(fig,'head.tif','-dtiff','-r70')   % tif figure output
print(fig,'head.eps','-depsc')  % vectorized figure output (publishing standard)
                               % it can be viewed using ghostscript, evince
                               % or by insterting it into word
savefig(fig,'fig.fig')    % save figure as fig file


fig2=figure;
title('the result for trigonometric functions');
subplot(2,1,1)
plot(x,y1,'r-x','linewidth',2,'displayname','cos(x)');hold on;
xlabel('x (m) for figure 1','FontSize',10,'FontWeight','bold')
ylabel('z (m) for figure 1','FontSize',12,'FontWeight','bold')
title('title for figure 1')
subplot(2,1,2)
title('title for figure 2')
plot(x,y2,'b:v','linewidth',3,'displayname','sin(x)');hold on;
xlabel('x (m) for figure 2','FontSize',10,'FontWeight','bold')
ylabel('z (m) for figure 2','FontSize',12,'FontWeight','bold')
title('title for figure 2')
savefig(fig2,'fig2.fig')

% save from GUI
% copy and paste lines in between figures

% MATLAB tutorial video is accessible from:
%https://www.youtube.com/watch?v=h-ld0Yj7zq4&t
%https://www.youtube.com/watch?v=cfXoisC7QDE

% To practis MATLAB, it is suggested to program the questions listed in the file below.
https://docs.google.com/document/d/18yqq0mHEVrBGt6cjJKKu6eYiTUh6duhc/edit?usp=sharing&ouid=110690160017580358063&rtpof=true&sd=true

