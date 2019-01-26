%% <<-- Archive -->>
%% Project: Acceleration of SVRG and Katyusha X by Inexact Preconditioning
%% Coded by: Fei Feng
%% Last update: 01/24/2019

classdef logistic
   properties
      data      % including A,b
      params    % hyperparameters
      n         % number of data
      p         % dimension of x
      min_value % minimum value
      M         % preconditioner
      diag_M
   end
   
   methods
      function prob = logistic(data, params)
          prob.data = data;
          prob.params = params;
          [prob.n, prob.p] = size(data.A);
          prob.min_value = 0;
          if (params.BUILD == 1)
              prob = buildPrecdn(prob);
          end
      end
      
      % compute gradient
      function g = grad(prob, x, batch_size)
          % full gradient
          if(batch_size == prob.n)
              temp = exp(prob.data.A*x.*(-prob.data.b));
              g = prob.data.A'*(temp.*(-prob.data.b)./(1+temp))/batch_size + 2*prob.params.LAMBDA2*x;
          % mini-batched gradient
          else
              y = randsample(prob.n, batch_size);
              sub_A(:,:)=prob.data.A(y,:);
              sub_b(:) = prob.data.b(y);
              sub_b = sub_b';
              temp = exp(sub_A*x.*(-sub_b));
              g = sub_A'*(temp.*(-sub_b)./(1+temp))/batch_size + 2*prob.params.LAMBDA2*x;
          end
      end
      
      % compute gradient difference 
      function g = scGradDiff(prob, w, x, batch_size)
          if(batch_size == prob.n)
              temp1 = exp(prob.data.A*w.*(-prob.data.b));
              temp2 = exp(prob.data.A*x.*(-prob.data.b));
              g = prob.data.A'*(temp1.*(-prob.data.b)./(1+temp1)-temp2.*(-prob.data.b)./(1+temp2))/batch_size + 2*prob.params.LAMBDA2*(w-x);
          else
              % randomly select #batch_size numbers from [n]
              y = randsample(prob.n,batch_size);
              sub_A(:,:)=prob.data.A(y,:);
              sub_b(:) = prob.data.b(y);
              sub_b = sub_b';
              temp1 = exp(sub_A*w.*(-sub_b));
              temp2 = exp(sub_A*x.*(-sub_b));
              g = sub_A'*(temp1.*(-sub_b)./(1+temp1)-temp2.*(-sub_b)./(1+temp2))/batch_size + 2*prob.params.LAMBDA2*(w-x);
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
                      w_block = w(block_start:block_end);
                      tilde_g_block = tilde_g(block_start:block_end);
                      % gradient descent
                      if(prob.params.BUILD == 0)
                          sub_A = prob.data.A(:, block_start:block_end);
                          temp = y_block - gamma * (1/(prob.n*eta) * sub_A' * (sub_A * (y_block-w_block))+tilde_g_block);
                      else
                          sub_M = prob.M(block_start:block_end, block_start:block_end);
                          temp = y_block - gamma * (sub_M/eta * (y_block-w_block)+tilde_g_block);
                      end
                      % proximal L1
                      for j = block_start:block_end
                          y(j) = sign(temp(j-block_start+1))*max(abs(temp(j-block_start+1))-gamma*lambda1, 0);
                      end
                  end
              end
          % preconditioner with FISTA    
          else
              if(prob.params.BUILD==0)
                  fprintf('use FISTA with params.BUILD=1');
              elseif(prob.params.M_BLOCK_SIZE>1)
                  t_old = 1;
                  x_old = w;
                  for iter = 1:prob.params.MAX_SUB_ITER
                      temp = y - gamma * (prob.M * (y-w)+tilde_g);
                      x_new(:)=sign(temp(:)).*(max(abs(temp(:))-gamma*lambda1, 0)); 
                      t_new = (1+sqrt(1+4*t_old^2))/2;
                      y = x_new' + (t_old-1)/t_new*(x_new'-x_old);
                      x_old = x_new';
                      t_old = t_new;
                  end
              else
                  temp = w - eta* tilde_g./prob.diag_M(:);
                  y(:)=sign(temp(:)).*(max(abs(temp(:))-eta*lambda1./prob.diag_M(:), 0)); 
              end
          end
      end
      
      % build preconditioner
      function prob = buildPrecdn(prob)
          block_size = prob.params.M_BLOCK_SIZE;
          block_num = ceil(prob.p/block_size);
          m = cell(block_num,1);
          square_b = spdiags(prob.data.b.^2,0,prob.n,prob.n);
          for i = 1:block_num
              block_start = 1+(i-1)*block_size;
              block_end = min(prob.p, i*block_size);
              [m{i}]= prob.data.A(:,block_start:block_end)'* square_b * prob.data.A(:,block_start:block_end)/(4*prob.n);
          end
          prob.M = blkdiag(m{:}) + prob.params.ALPHA * speye(prob.p);
          if(block_size == 1)
              prob.diag_M = diag(prob.M);
          end
      end
      
      % sub-optimality
      function error = checkError(prob, x)
          error = 1/prob.n * norm(log(1+exp(-prob.data.b.*(prob.data.A*x))),1) + prob.params.LAMBDA1 * norm(x,1)+prob.params.LAMBDA2*norm(x)^2 - prob.min_value;
      end
   end
end