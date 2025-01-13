clc
clear
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOAD DATASET + EDA + create train test matrix%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filename = 'dataset/ratings.csv';
ratings_table = readtable(filename);

head(ratings_table);

userId = ratings_table.userId;
movieId = ratings_table.movieId;
rating = ratings_table.rating;

disp('Unique values for ratings:'); 
disp(unique(rating));

unique_users = unique(userId); 
unique_movies = unique(movieId);  

[~, userId_mapped] = ismember(userId, unique_users); %assegna un indice consecutivo a ciascun ID 
[~, movieId_mapped] = ismember(movieId, unique_movies); %assegna un indice consecutivo a ciascun ID 

n_users = numel(unique_users);  
n_movies = numel(unique_movies);

disp(n_users)
disp(n_movies)

ratings_matrix = sparse(userId_mapped, movieId_mapped, rating, n_users, n_movies);

n_movies=500;
subset_ratings_matrix = ratings_matrix(:, 1:n_movies);
users_with_ratings = sum(subset_ratings_matrix > 0, 2)>=5;

filtered_ratings_matrix = subset_ratings_matrix(users_with_ratings, :);
dense_ratings_matrix = full(filtered_ratings_matrix);
disp('First 10 rows and first 10 col of ratings matrix:');
disp(dense_ratings_matrix(1:10, 1:10)); 

test_ratios = [0.1, 0.3, 0.7];
split_data = struct();


for i = 1:length(test_ratios) 
    test_ratio = test_ratios(i);
    [train_matrix, test_matrix] = dataset_splitting(filtered_ratings_matrix, test_ratio);
    
    split_data(i).test_ratio = test_ratio;
    split_data(i).train_matrix = train_matrix;
    split_data(i).test_matrix = test_matrix;
  
    disp(['Test ratio: ', num2str(test_ratio)]);
    disp(['Number of ratings in Training matrix: ', num2str(nnz(train_matrix))]);
    disp(['Number of ratings in Test matrix: ', num2str(nnz(test_matrix))]);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Nuclear Norm Minimization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


resultsNN = struct();

for i = 1:length(test_ratios)
    [completedMatrixNN, timeNN] = nuclear_norm_minimization(split_data(i).train_matrix);
    completedMatrixNN = normalize_matrix(completedMatrixNN, 0.5, 5, 0.5);
    nmseNN = calculate_NMSE(split_data(i).test_matrix, completedMatrixNN);
    resultsNN(i).test_ratio = test_ratios(i);
    resultsNN(i).nmse = nmseNN;
    resultsNN(i).time = timeNN;
    disp(['Test Ratio: ', num2str(test_ratios(i))]);
    disp(['Nuclear Norm NMSE: ', num2str(nmseNN)]);
    disp(['Execution Time: ', num2str(timeNN)]);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SVT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
max_iter=2000;
tau_values=[1700, 2000];
delta_values=[0.5, 1];

resultsSVT = struct();

for i = 1:length(test_ratios)
    [best_params, ~] = svt_grid_search(split_data(i).train_matrix, tau_values, delta_values, max_iter);
    tau=best_params.tau;
    delta= best_params.delta;
    [completed_matrixSVT, timeSVT, error_history] = svt(split_data(i).train_matrix, filtered_ratings_matrix, tau, delta, max_iter);
    completed_matrixSVT = normalize_matrix(completed_matrixSVT, 0.5, 5, 0.5);
    nmseSVT = calculate_NMSE(split_data(i).test_matrix, completed_matrixSVT);
    resultsSVT(i).test_ratio = test_ratios(i);
    resultsSVT(i).nmse = nmseSVT;
    resultsSVT(i).time = timeSVT;
    resultsSVT(i).tau = tau;
    resultsSVT(i).delta=delta;
    resultsSVT(i).error_history = error_history; 

    disp(['Test Ratio: ', num2str(test_ratios(i))]);
    disp(['SVT NMSE: ', num2str(nmseSVT)]);
    disp(['Execution Time: ', num2str(timeSVT.total)]);
end


figure;
hold on;
for i = 1:length(test_ratios)
    plot(1:length(resultsSVT(i).error_history), resultsSVT(i).error_history, '-', 'LineWidth', 1.5, ...
        'DisplayName', ['Test Ratio: ', num2str(test_ratios(i))]);
end
hold off;
xlabel('iter');
ylabel('NMSE (dB)');
title('NMSE SVT for different test ratio ');
legend('show');
grid on;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Explicit GD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
lr_values= [0.0001, 0.001, 0.01];
r_values=[2, 5, 10];
lambda_values = [0.01, 0.1, 1];

resultsGD = struct();
for i = 1:length(test_ratios)
    [best_params, ~] = gd_grid_search(split_data(i).train_matrix, r_values, lambda_values, lr_values, max_iter);
    r=best_params.r;
    lambda=best_params.lambda;
    lr=best_params.lr;
    [completed_matrixGD, timeGD, error_history] = mf_explicit_gd(split_data(i).train_matrix, filtered_ratings_matrix, r, lambda, max_iter, lr);
    completed_matrixGD = normalize_matrix(completed_matrixGD, 0.5, 5, 0.5);
    nmseGD = calculate_NMSE(split_data(i).test_matrix, completed_matrixGD);
    resultsGD(i).test_ratio = test_ratios(i);
    resultsGD(i).r = r;
    resultsGD(i).lambda = lambda;
    resultsGD(i).lr = lr;
    resultsGD(i).nmse = nmseGD;
    resultsGD(i).time = timeGD.total;
    resultsGD(i).error_history = error_history;

    disp(['Test Ratio: ', num2str(test_ratios(i))]);
    disp(['GD NMSE: ', num2str(nmseGD)]);
    disp(['Execution Time: ', num2str(timeGD.total)]);
end

figure;
hold on;
for i = 1:length(test_ratios)
    plot(1:length(resultsGD(i).error_history), resultsGD(i).error_history, '-', 'LineWidth', 1.5, ...
        'DisplayName', ['Test Ratio: ', num2str(test_ratios(i))]);
end
hold off;
xlabel('iter');
ylabel('NMSE (dB)');
title('GD NMSE for different test ratio with explicit feedback');
legend('show');
grid on;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Explicit ALS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
max_iter=1000;
r_values=[2, 5, 10];
lambda_values = [ 0.1,  1];

resultsALS = struct();
for i = 1:length(test_ratios)
    [best_params, ~] = als_grid_search(split_data(i).train_matrix, r_values, lambda_values, max_iter);
    r=best_params.r;
    lambda=best_params.lambda;
    [completed_matrixALS, timeALS, error_history] = mf_explicit_als(split_data(i).train_matrix, filtered_ratings_matrix, r, lambda,  max_iter);
    completed_matrixALS = normalize_matrix(completed_matrixALS, 0.5, 5, 0.5);
    nmseALS = calculate_NMSE(split_data(i).test_matrix, completed_matrixALS);
    resultsALS(i).test_ratio = test_ratios(i);
    resultsALS(i).r = r;
    resultsALS(i).lambda = lambda;
    resultsALS(i).nmse = nmseALS;
    resultsALS(i).time = timeALS.total;
    resultsALS(i).error_history = error_history;

    disp(['Test Ratio: ', num2str(test_ratios(i))]);
    disp(['ALS NMSE: ', num2str(nmseALS)]);
    disp(['Execution Time: ', num2str(timeALS.total)]);
end

figure;
hold on;
for i = 1:length(test_ratios)
    plot(1:length(resultsALS(i).error_history), resultsALS(i).error_history, '-', 'LineWidth', 1.5, ...
        'DisplayName', ['Test Ratio: ', num2str(test_ratios(i))]);
end
hold off;
xlabel('iter');
ylabel('NMSE (dB)');
title('ALS NMSE for different Test Ratio with explicit feedback');
legend('show');
grid on;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Implicit GD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
lr_values = [0.000001];
r_values = [10 , 15];
lambda_values = [0.01, 0.1, 0.5];
alpha_values= 40;
resultsGD_implicit = struct();

for i = 1:length(test_ratios)
    [best_params, ~] = gd_implicit_grid_search(split_data(i).train_matrix, r_values, lambda_values, lr_values,alpha_values, max_iter);
    r=best_params.r;
    alpha=best_params.alpha;
    lambda=best_params.lambda;
    lr=best_params.lr;
    [completed_matrixGD, timeGD,  error_history] = mf_implicit_gd(split_data(i).train_matrix, filtered_ratings_matrix, r, lambda,  max_iter, lr, alpha);
    completed_matrixGD = normalize_matrix(completed_matrixGD, 0.5, 5, 0.5);
    nmseGD = calculate_NMSE(split_data(i).test_matrix, completed_matrixGD);
    resultsGD_implicit(i).test_ratio = test_ratios(i);
    resultsGD_implicit(i).r = r;
    resultsGD_implicit(i).lambda = lambda;
    resultsGD_implicit(i).lr = lr;
    resultsGD_implicit(i).alpha = alpha;
    resultsGD_implicit(i).nmse = nmseGD;
    resultsGD_implicit(i).time = timeGD.total;
    resultsGD_implicit(i).error_history = error_history;
    
    disp(['Test Ratio: ', num2str(test_ratios(i))]);
    disp(['Implicit GD NMSE on test: ', num2str(nmseGD)]);
    disp(['Execution Time: ', num2str(timeGD.total)]);
end

figure;
hold on;
for i = 1:length(test_ratios)
    plot(1:length(resultsGD_implicit(i).error_history), resultsGD_implicit(i).error_history, '-', 'LineWidth', 1.5, ...
        'DisplayName', ['Test Ratio: ', num2str(test_ratios(i))]);
end
hold off;
xlabel('iter');
ylabel('NMSE (dB)');
title('GD NMSE for different test ratio with implicit feedback');
legend('show');
grid on;


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Nuclear Norm Minimization as SDP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%{ 
resultsSDP = struct();

for i = 1:length(test_ratios)
    [completedMatrixSDP, timeSDP] = sdp_nuclear_norm_minimization(split_data(i).train_matrix);
    completedMatrixSDP = normalize_matrix(completedMatrixSDP, 0.5, 5, 0.5);
    nmseSDP = calculate_NMSE(split_data(i).test_matrix, completedMatrixSDP);
    resultsSDP(i).test_ratio = test_ratios(i);
    resultsSDP(i).nmse = nmseSDP;
    resultsSDP(i).time = timeSDP;
    disp(['Test Ratio: ', num2str(test_ratios(i))]);
    disp(['Nuclear Norm NMSE: ', num2str(nmseSDP)]);
    disp(['Execution Time: ', num2str(timeSDP)]);
end
%}
%%
