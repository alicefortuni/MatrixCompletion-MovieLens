function normalized_matrix = normalize_matrix(matrix, minValue, maxValue, step)
    normalized_matrix = round(matrix/ step) * step;
    normalized_matrix = max(minValue, min(maxValue, normalized_matrix));
end