function selected_features = select_features(features, labels)
    % 计算特征与标签的相关性
    corr_values = zeros(size(features, 2), 1);
    for i = 1:size(features, 2)
        corr_values(i) = abs(corrcoef(features(:,i), double(labels))(1,2));
    end
    
    % 选择相关性最高的前30个特征
    [~, idx] = sort(corr_values, 'descend');
    selected_features = features(:, idx(1:min(30, length(idx))));
end