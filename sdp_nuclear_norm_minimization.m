function [X, timing] = sdp_nuclear_norm_minimization(M)
    [n_users, n_movies] = size(M);
    omega = find(M);
    tic;
    cvx_begin sdp
        variable X(n_users, n_movies) 
        variable W1(n_users, n_users) semidefinite 
        variable W2(n_movies, n_movies) semidefinite

        minimize(trace(W1) + trace(W2))

        subject to  
            X(omega) == M(omega)
            [W1, X; X', W2] >= 0;
    cvx_end
    timing=toc;

end
