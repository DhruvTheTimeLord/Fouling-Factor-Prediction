function [data_clean] = outlier_detection_and_replacement(data)
% This function detects and replaces outliers in a dataset using the median absolute deviation (MAD) method.

% Define outlier threshold as a multiple of MAD
threshold = 3;

% Calculate median and MAD
median_data = median(z);
mad = median(abs(z - median_data));

% Identify outlier indices
outlier_indices = find(abs(z - median_data) > threshold*mad);

% Replace outliers with median
data_clean = zeros();
data_clean(outlier_indices) = ones();
data_clean=data_clean';

end
