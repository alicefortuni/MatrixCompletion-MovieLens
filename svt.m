function [M, timing, error_history] = svt(X, original_matrix, tau, delta, max_iter)

    X = full(X);
    omega = X > 0;
    [n_users, n_movies] = size(X);
    rng(2)
    M =  zeros(n_users,n_movies);
    Y = zeros(n_users, n_movies);
    
    epsilon = 1e-4; 
    error_history = zeros(max_iter, 1);
    tic;
   
    for k = 1:max_iter 
        
        [U, S, V] = svd(Y, 'econ');
        S = diag(max(diag(S) - tau, 0));
        M = U * S * V';      
       
        residual = omega.*(X - M); 

        Y = Y + delta * residual;
           
        error_history(k) = calculate_NMSE(original_matrix, M);
        
        %loss for convergence
        current_loss=tau*sum(diag(S))+0.5*norm(M, 'fro')^2;
        if k > 1
            delta_loss= abs(current_loss - previous_loss) / abs(previous_loss);
            if delta_loss < epsilon
                fprintf('Convergence reached after %d iterations\n', k);
                error_history =error_history(1:k);
            break;
            end
        end
        previous_loss=current_loss;
        
    end
    timing.total = toc;
end
