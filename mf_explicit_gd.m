function [M, timing,  error_history] = mf_explicit_gd(X,original_matrix, r, lambda, max_iter, lr)

    [n_users,n_movies] = size(X);
    rng(20);
    A = rand(n_users,r);
    B = rand(n_movies, r);
    
    error_history = zeros(max_iter, 1);
    
    X = full(X);
    mask = X > 0;
    epsilon=1e-4; 
    tic;
    
     for iter = 1:max_iter

        for i = 1:n_users
            for j = 1:n_movies
                if mask(i,j) 
                    e_ij  =  dot(A(i,:), B(j,:)) - X(i,j) ; 
                    for k = 1:r
                        A(i, k) = A(i, k) - 2 *lr * ( e_ij * B(j, k) +  lambda * A(i, k));
                        B(j, k) = B(j, k) - 2 *lr * ( e_ij * A(i, k) +  lambda * B(j, k));
                    end
                    
                end
            end
        end

        M = A * B';
        error_history(iter) = calculate_NMSE(original_matrix, M);
        current_loss = 0;
        %convergence based on loss
        for i = 1:n_users
            for j = 1:n_movies
                if mask(i,j) 
                    e_ij  =  dot(A(i,:), B(j,:)) - X(i,j) ; 
                    current_loss = current_loss + e_ij^2;
                end
            end
         end
        
        current_loss = current_loss + lambda * (norm(A, 'fro')^2 + norm(B, 'fro')^2);
        if iter > 1
            delta_loss = abs(current_loss - previous_loss)/ abs(previous_loss);
            if delta_loss < epsilon
                fprintf('Convergece reached after %d iter\n', iter);
                error_history =error_history(1:iter);
                break;
            end
        end
        previous_loss = current_loss;
 
     end
       
    timing.total = toc;
end