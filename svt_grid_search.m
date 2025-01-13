function [best_params, error_grid] = svt_grid_search(train_matrix, tau_values, delta_values, max_iter)

    [M_train, M_validation]=dataset_splitting(train_matrix, 0.2);

    num_tau = length(tau_values);
    num_delta = length(delta_values);
    error_grid = zeros(num_tau, num_delta);

    best_error = Inf;
    best_params = struct('tau', [], 'delta', []);

   
    for t = 1:num_tau
        for d = 1:num_delta
            tau = tau_values(t);
            delta = delta_values(d);

            [completedMatrix, ~] = svt(M_train, train_matrix, tau, delta, max_iter);

            nmse = calculate_NMSE(M_validation, completedMatrix); 
            error_grid(t, d) = nmse;

            if nmse < best_error
                best_error = nmse;
                best_params.tau = tau;
                best_params.delta = delta;
            end
        end
    end

   %{
    fprintf('Miglior NMSE: %.4f dB\\n', best_error);
    fprintf('Parametri ottimali: tau=%.2f, delta=%.2f\\n', best_params.tau, best_params.delta);
   %}
end
