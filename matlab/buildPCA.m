%% <<-- Archive -->>
%% Project: Acceleration of SVRG and Katyusha X by Inexact Preconditioning
%% Coded by: Fei Feng
%% Last update: 01/24/2019

function data = buildPCA
n = 2000;
p = 100;
DSTYLE = 1;
DELTA = 100;
K = 10;
if(mod(p,2) ~= 0)
    fprintf('the dimension of PCA should be an even number.')
    return;
end
data.A_group = normc(rand(p, n));
for i=1:p
  vec = zeros(p, 1);
  vec(i) = 5*i;
  data.A_group(:,i) = data.A_group(:,i)+vec;
end

data.A = zeros(p,p);
for i = 1:n
    data.A = data.A + data.A_group(:,i)*data.A_group(:,i)';
end
data.A = data.A/n;
data.b = normc(rand(p, 1));
if(DSTYLE == 1)
    data.D_group = DELTA * ones(p, n);
    data.D_group(:,1:n/2) = -DELTA;
else
    data.D_group = K/(n-1) * ones(p,n);
    y = randi(n,p,1);
    for i = 1:p
        data.D_group(i,y(i))= -K;
    end
end
end