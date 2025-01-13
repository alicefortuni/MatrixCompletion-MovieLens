function [M,timing, error_history] = mf_explicit_als(X, original_matrix, r, lambda,  max_iter)

    [n_users, n_movies] = size(X);

    rng(1);
    A = rand(n_users, r)*100; 
    B = rand(n_movies, r)*100;
    error_history = zeros(max_iter, 1);
    X = full(X);
    mask = (X > 0);
    epsilon=1e-4;
    tic;
    
    for iter = 1:max_iter
        for i = 1:n_users
            indices = find(mask(i, :)); 
            B_i = B(indices, :); %matrice vettori latenti : Omega_i x r
            X_i = X(i, indices)'; %vettore colonna: Omega_i x 1
            A(i, :) = (B_i' * B_i + lambda * eye(r)) \ (B_i' * X_i);
        end

        for j = 1:n_movies
            indices = find(mask(:, j));
            A_j = A(indices, :); 
            X_j = X(indices, j); 
            B(j, :) = (A_j' * A_j + lambda * eye(r)) \ (A_j' * X_j);
        end

        M = A * B';
   
        
        %loss for convergence 
        current_loss=0;
        for i = 1:n_users
            for j = 1:n_movies
                if mask(i,j) 
                    e_ij  =  dot(A(i,:), B(j,:)) - X(i,j) ; 
                    current_loss = current_loss + e_ij^2;
                end
            end
        end
        error_history(iter) = calculate_NMSE(original_matrix, M);
        
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
