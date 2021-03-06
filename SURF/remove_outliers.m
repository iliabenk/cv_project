function [match_points_location_cleaned] = remove_outliers(data, min_inliers_percent, data_type)

min_area = 10 ^ 8;
best_inliers_bitmask = [];

if strcmp(data_type, 'pairs')
    match_points_location = data.I_test_vaild_points(data.test_points(:, 2));
    match_points_location = match_points_location.Location;
else
   match_points_location = data; 
end

while isempty(best_inliers_bitmask)
    for iteration = 1 : 20
        random_indices = randperm(size(match_points_location, 1));
        random_indices = random_indices(1 : 2);

        random_match_points = match_points_location(random_indices, :);

        min_x = min(random_match_points(:, 1));
        min_y = min(random_match_points(:, 2));
        max_x = max(random_match_points(:, 1));
        max_y = max(random_match_points(:, 2));

        inliers_bitmask = (match_points_location(:, 1) >= min_x - 100) .* (match_points_location(:, 1) <= max_x + 100) .* (match_points_location(:, 2) >= min_y - 100) .* (match_points_location(:, 2) <= max_y + 100);

        inliers_percent = sum(inliers_bitmask) / size(match_points_location, 1);

        if inliers_percent >= min_inliers_percent
            area = (max_x - min_x) * (max_y - min_y);

            if area < min_area
               min_area = area; 
               best_inliers_bitmask = inliers_bitmask;
            end
        end
    end
end

% best_inliers_index = best_inliers_bitmask .* [1 : length(best_inliers_bitmask)].';
if strcmp(data_type, 'pairs')
    match_points_location_cleaned = data.test_points(best_inliers_bitmask == 1, :); 
else
    match_points_location_cleaned = data(best_inliers_bitmask == 1, :); 
end
end


