function [train_matrix, test_matrix]=dataset_splitting(M, test_ratio)

    [n_users,n_movies] = size(M);
    train_matrix = sparse(n_users, n_movies);
    test_matrix = sparse(n_users, n_movies); 

    rng(5);

    for user = 1:n_users 
        rated_movies = find(M(user, :));
        n_ratings = length(rated_movies);

        if n_ratings > 0
            n_test = round(n_ratings * test_ratio); 
            n_test = min(n_test, n_ratings - 1); 
            
            perm = randperm(n_ratings);

            test_idx = rated_movies(perm(1:n_test));
            train_idx = rated_movies(perm(n_test+1:end));
            test_matrix(user, test_idx) = M(user, test_idx);
            train_matrix(user, train_idx) = M(user, train_idx);

        else
        warning('User %d has no ratings in the original dataset', user);
        end
        
    end
    assert(nnz(train_matrix) + nnz(test_matrix) == nnz(M), 'Total number of ratings mismatch');
end
