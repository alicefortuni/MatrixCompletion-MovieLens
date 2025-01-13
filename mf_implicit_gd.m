function [X, timing,  error_history] = mf_implicit_gd(M, original_matrix, r, lambda,  max_iter, lr, alpha)
    [n_users, n_movies] = size(M);
    
    rng(25); 
    A = rand(n_users, r);
    B = rand(n_movies, r); 
    M=M/5;
    M=full(M);
    Phi = double(M > 0);
    C = 1 + alpha * M;  
    epsilon = 1e-4; 
    error_history = zeros(max_iter, 1);
    tic;
    
    for iter = 1:max_iter
        for i = 1:n_users
            for j = 1:n_movies
                    e_ij = Phi(i, j) - dot(A(i, :), B(j, :));
                    for k = 1:r
                        A(i, k) = A(i, k) + 2 * lr * (C(i, j) * e_ij * B(j, k) - lambda * A(i, k));
                        B(j, k) = B(j, k) + 2 * lr * (C(i, j) * e_ij * A(i, k) - lambda * B(j, k));
                    end               
            end
        end

        X = (A * B')*5;
        error_history(iter) = calculate_NMSE(original_matrix, X);

        %loss for convergence
        
        current_loss = 0;
        for i = 1:n_users
            for j = 1:n_movies
                e_ij=C(i, j) * (Phi(i, j) - dot(A(i, :), B(j, :)))^2;
                current_loss = current_loss + e_ij;
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
