function [X, timing] = nuclear_norm_minimization(M) 
    [n_users, n_movies] = size(M);
    omega = find(M);

    tic; 
    cvx_begin quiet
        variable X(n_users, n_movies)
        minimize(norm_nuc(X))
        subject to
            X(omega) == M(omega)
    cvx_end
    timing = toc;
end
   
