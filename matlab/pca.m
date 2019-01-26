%% <<-- Archive -->>
%% Project: Acceleration of SVRG and Katyusha X by Inexact Preconditioning
%% Coded by: Fei Feng
%% Last update: 01/24/2019

classdef pca
   properties
      data      % including A,b,A_group,D_group
      params    % hyperparameters
      n         % number of data
      p         % dimension of variable
      min_value % minimum value
      M         % preconditioner
   end
   
   methods
      function prob = pca(data,params)
          prob.data = data;
          prob.params = params;
          [prob.p, prob.n] = size(data.A_group);
          prob.min_value = 0; 
          if (params.PRECDN == 1 && params.BUILD == 1)
              prob = buildPrecdn(prob);
          end
      end
      
      % compute gradient
      function g = grad(prob, x, batch_size)
          % full gradient
          if(batch_size == prob.n)
              g = prob.data.A * x + prob.data.b;
          % mini-batched gradient
          else
              y = randsample(prob.n, batch_size);
              g = zeros(prob.p,1);
              for i = 1:batch_size
                  g = g + dot(prob.data.A_group(:,y(i)),x)*prob.data.A_group(:,y(i))+prob.data.D_group(:,y(i)).*x;
              end
              g = g/batch_size+prob.data.b;
          end
      end
      
      % compute gradient difference 
      function g = scGradDiff(prob, w, x, batch_size)
          if(batch_size == prob.n)
              g = prob.data.A*(w-x);
          else
              % randomly select #batch_size numbers from [n]
              y = randsample(prob.n,batch_size);
              g = zeros(prob.p,1);
              for i = 1:batch_size
                  g = g + dot(prob.data.A_group(:,y(i)),w-x)*prob.data.A_group(:,y(i))+prob.data.D_group(:,y(i)).*(w-x);
              end
              g = g/batch_size;
          end
      end
      
      % proximal with block diagonal preconditioner
      function y = blockDiagonalProx(prob, w, tilde_g)
          % set initial point as w
          y = w;       
          gamma = prob.params.GAMMA;
          lambda1 = prob.params.LAMBDA1;
          eta = prob.params.ETA;
          % no preconditioner
          if(prob.params.PRECDN == 0)
              % gradient descent
              x=w-eta*tilde_g;
              % proximal L1
              y(:)=sign(x(:)).*(max(abs(x(:))-eta*lambda1, 0));             
          % preconditioner with BCD
          elseif(prob.params.FISTA == 0)
              block_size =  prob.params.BCD_SIZE;
              block_num = ceil(prob.p / block_size);
              for iter = 1:prob.params.MAX_SUB_ITER
                  for i = 1:block_num
                      % index info
                      block_start = 1+(i-1)*block_size;
                      block_end = min(prob.p, i*block_size);
                      y_block = y(block_start:block_end);
                      tilde_g_block = tilde_g(block_start:block_end);
                      % gradient descent
                      sub_M = prob.M(block_start:block_end,:);
                      temp = y_block - gamma * (sub_M/eta * (y-w)+tilde_g_block);
                      % proximal L1
                      for j = block_start:block_end
                          y(j) = sign(temp(j-block_start+1))*max(abs(temp(j-block_start+1))-gamma*lambda1, 0);
                      end
                  end
              end
          % preconditioner with FISTA    
          else
              if(prob.params.M_BLOCK_SIZE>1)
                  t_old = 1;
                  x_old = w;
                  for iter = 1:prob.params.MAX_SUB_ITER
                      temp = y - gamma * (prob.M/eta * (y-w)+tilde_g);
                      x_new(:)=sign(temp(:)).*(max(abs(temp(:))-gamma*lambda1, 0)); 
                      t_new = (1+sqrt(1+4*t_old^2))/2;
                      y = x_new' + (t_old-1)/t_new*(x_new'-x_old);
                      x_old = x_new';
                      t_old = t_new;
                  end
              else
                  temp = w - eta*tilde_g./prob.M;
                  y(:)=sign(temp(:)).*(max(abs(temp(:))-(eta*lambda1)./prob.M(:), 0)); 
              end
          end
      end
      
      % build preconditioner
      function prob = buildPrecdn(prob)
          if(prob.params.M_BLOCK_SIZE == 1)
              prob.M = diag(prob.data.A) + prob.params.ALPHA*ones(prob.p,1);
          else
              block_size = prob.params.M_BLOCK_SIZE;
              block_num = ceil(prob.p/block_size);
              m = cell(block_num,1);
              for i = 1:block_num
                  block_start = 1+(i-1)*block_size;
                  block_end = min(prob.p, i*block_size);
                  [m{i}]= prob.data.A(block_start:block_end,block_start:block_end);
              end
              prob.M = blkdiag(m{:}) + prob.params.ALPHA*speye(prob.p);
          end         
      end
      
      % sub-optimality
      function error = checkError(prob, x)
          error = 0.5*x'*prob.data.A*x+ prob.data.b'*x + prob.params.LAMBDA1* norm(x,1)-prob.min_value;
      end
   end
end