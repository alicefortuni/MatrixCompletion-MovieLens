function nmse = calculate_NMSE(test_matrix, predicted_matrix)     
    [rows_test, cols_test] = find(test_matrix); 
    real_values = test_matrix(sub2ind(size(test_matrix), rows_test, cols_test)); 
    predictedValues = predicted_matrix(sub2ind(size(predicted_matrix), rows_test, cols_test));

    numerator = sum((real_values - predictedValues).^2);
    denominator = sum(real_values.^2);
    nmse = 10*log10(numerator/denominator);
end