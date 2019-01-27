%% <<-- Archive -->>
%% Project: Acceleration of SVRG and Katyusha X by Inexact Preconditioning
%% Authors: Yanli Liu, Fei Feng, Wotao Yin
%% Last update: 01/24/2019

function InexactPrecdnTest(data)
%% Regulator Parameters
params.LAMBDA1 = 1e-3;          % l1-regularization parameter
params.LAMBDA2 = 0;             % l2-regularization parameter

%% Algorithm Parameters
params.VERBOSE = 1;             % choose 1 to print details
params.MAX_EPOCH = 100000000;   % max number of outer loop in SVRG
params.MAX_ITER = 100;          % max number of inner loop in SVRG
params.CHECK_STEP = 200;        % number of iterations to check
params.TOL = 1e-10;             % accuracy tolerance
params.BATCH_SIZE = 1;          % batch-size for stochastic gradient
params.KATYUSHA = 1;            % choose 1 to run Katyusha, else Prox-SVRG
params.TAU = 0.45;              % Katyusha-X: momentum weight, tau=0.5 is PSVRG

%% Stepsize Parameters
params.ETA = 0.015;             % stepsize for SVRG smooth term

%% Preconditioner Parameters
params.PRECDN = 1;              % choose 1 to use preconditioner
params.FISTA = 1;               % choose 1 to use FISTA; 0 to use BCD
params.MAX_SUB_ITER =20;        % max number of iterations for subproblem
params.BUILD = 1;               % choose 1 to formulate preconditioner
params.BCD_SIZE = 1;            % block-size for BCD
params.M_BLOCK_SIZE = 1;        % block-size for preconditioner
params.GAMMA = 0;               % stepsize for subproblem
params.ALPHA = 15;              % preconditioner = \alpha I + 1/n~(A^TA)

%% Problem settings
prob = lasso(data, params);
%prob = logistic(data, params);
%prob = pca(data,params);    
prob.min_value = -0.0983942174; % change min_value for different problem and dataset.
x = zeros(prob.p,1);  
w = zeros(prob.p,1);
if(params.KATYUSHA)
    y_old = zeros(prob.p,1);
    y_new = zeros(prob.p,1);
end

%% Inexact Preconditioner Solver
fprintf('\nCalling Inexact Preconditioner Solver_MATLAB 01/24/2018\n');
fprintf('-----------------------------------------------\n');
if(params.VERBOSE)
    fprintf('PRECDN = %d\n', params.PRECDN);
    fprintf('KATYUSHA = %d\n', params.KATYUSHA);
    fprintf('ETA = %d\n', params.ETA);
    fprintf('TAU = %d\n', params.TAU);
    fprintf('GAMMA = %d\n', params.GAMMA);
    fprintf('ALPHA = %d\n', params.ALPHA);
end
fprintf('Time\t,Epoch\t,Error\n');
tic
for i = 0:params.MAX_EPOCH
    
    % check sub-optimality
    if(mod(i,params.CHECK_STEP)==0)
        time = toc;
        error = checkError(prob, x);
        if(params.VERBOSE)
            fprintf('%.3f,%d,%.10f\n', time, i, error);
        end
        if(abs(error) < params.TOL)
            break;
        end
        tic
    end
    
    % Nesterov acceleration
    if(params.KATYUSHA)
        y_old = y_new;
        y_new = w;
        x = (1.5*y_new + 0.5*x - (1-params.TAU)*y_old)/(1+params.TAU);
    else
        x = w;
    end
    
    % SVRG 
    g = grad(prob, x, prob.n);
    for j = 1:params.MAX_ITER
        % a variance-reduced stochastic gradient
        tilde_g = g + scGradDiff(prob, w, x, params.BATCH_SIZE);
        % solve non-smooth part
        w = blockDiagonalProx(prob, w, tilde_g); 
    end
end





