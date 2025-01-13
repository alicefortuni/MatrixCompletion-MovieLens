function [best_params, error_grid] = als_grid_search(train_matrix, r_values, lambda_values, max_iter)

    [M_train, M_validation]=dataset_splitting(train_matrix, 0.2);
    
    error_grid = zeros(length(r_values), length(lambda_values));    
    best_error = Inf;
    
    best_params = struct('r', [], 'lambda', []);

    for i = 1:length(r_values)
        for j = 1:length(lambda_values)
            r = r_values(i);
            lambda = lambda_values(j);

            [completedMatrix, ~] = mf_explicit_als(M_train, train_matrix,  r,  lambda,  max_iter);

            nmse = calculate_NMSE(M_validation, completedMatrix);
            error_grid(i, j) = nmse;

            if nmse < best_error
                best_error = nmse;
                best_params.r = r;
                best_params.lambda = lambda;
            end
        end
    end
end