function [best_params, error_grid] = gd_implicit_grid_search(train_matrix, r_values, lambda_values, lr_values, alpha_values, max_iter)
    
    [M_train, M_validation]=dataset_splitting(train_matrix, 0.2);

    error_grid = zeros(length(r_values), length(lambda_values), length(lr_values), length(alpha_values));
    
    best_error = Inf;
    best_params = struct('r', [], 'lambda', [], 'lr', []);
    for i = 1:length(r_values)
        for j = 1:length(lambda_values)
            for k = 1:length(lr_values)
                 for m = 1:length(alpha_values)
                    r = r_values(i);
                    lambda = lambda_values(j);
                    lr = lr_values(k);
                    alpha=alpha_values(m);
                    [completedMatrix, ~] = mf_implicit_gd(M_train,train_matrix, r, lambda, max_iter, lr, alpha);

                    nmse = calculate_NMSE(M_validation, completedMatrix);
                    error_grid(i, j, k, m) = nmse;

                    if nmse < best_error
                        best_error = nmse;
                        best_params.r = r;
                        best_params.lambda = lambda;
                        best_params.lr = lr;
                        best_params.alpha = alpha;
                    end
                 end
            end
        end
    end
end