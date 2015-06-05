pwd
ls
help pwd
3-5
cos(2*3.14)
a=3-5
b=cos(2*pi)
a-b
c=a-b
a=a-b
d-b
a-b;
d=.6
b-d*c/b*a^b

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
e=[4
5
6
7];
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
e(0); %base 1
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

e( e>5  &&  e <=7)=6.5

e( e<1.2 || e>8.5)=11.2

save('a.mat')

clear
d
e

load('a.mat')
d
e

% use struct to load a workspace
st=load('a.mat')

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
 
if index1>0
  a=2;
  b=sqrt(a);
    if a>5
       d=7;
    end
elseif index==0
  a=4;
  b=a^2;
else    
  a=3;
end
a=1;
%% ---- end of understanding if statements ----




%% --------understanding loops -------------------
clear
% loops
for i=1:10
   j=i     % no semicolon for displaying purpose
end


% nested loops
for i=1:10
 for j=1:5
     k=k+i-j;
 end
end
%% -----------end of understanding loops----------



% ----- understanding stack -----


% ----- three different method doing the same function ---
% the first
clear
b=0;
c=0;
tic
while b<101
   c=c+b; 
   b=b+1;
end
toc
% the second
clear
c=0;
tic
for i=1:100
c=c+i;
end
toc
%the third
clear
tic
c=1:100;
d=sum(c);
toc
