%% <<-- Archive -->>
%% Project: Acceleration of SVRG and Katyusha X by Inexact Preconditioning
%% Authors: Yanli Liu, Fei Feng, Wotao Yin
%% Last update: 01/24/2019

function data = buildPCA
n = 2000; % number of individual functions
p = 100;  % dimension of x
DELTA = 100;
if(mod(p,2) ~= 0)
    fprintf('the dimension should be an even number.')
    return;
end

% A_group = [a_1,a_2,...,a_n]
data.A_group = normc(rand(p, n)); 
% modify A_group to [c_1,c_2,...,c_n]
for i=1:p
  vec = zeros(p, 1);
  vec(i) = 5*i;
  data.A_group(:,i) = data.A_group(:,i)+vec;
end

% formulate matrix A:=1/n (\sum_i c_ic_i'+D_i), 
% since \sum D_i = 0, we don't consider D_i at this moment
data.A = zeros(p,p);
for i = 1:n
    data.A = data.A + data.A_group(:,i)*data.A_group(:,i)';
end
data.A = data.A/n;

% formulate vector b
data.b = normc(rand(p, 1));

% formulate D_1, ..., D_n
data.D_group = DELTA * ones(p, n);
data.D_group(:,1:n/2) = -DELTA;

end