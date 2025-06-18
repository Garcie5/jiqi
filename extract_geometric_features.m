function geo_features = extract_geometric_features(img_binary)
    % 提取水珠区域的几何特征
    stats = regionprops(img_binary, 'Area', 'Perimeter', 'EquivDiameter', ...
                        'Eccentricity', 'Solidity', 'MajorAxisLength');
    
    if isempty(stats)
        % 无水珠区域时返回零向量
        geo_features = zeros(1, 8);
        return;
    end
    
    % 计算特征统计量
    areas = [stats.Area];
    perimeters = [stats.Perimeter];
    diameters = [stats.EquivDiameter];
    eccentricities = [stats.Eccentricity];
    solidities = [stats.Solidity];
    major_axes = [stats.MajorAxisLength];
    
    % 形状因子（圆形度）
    shape_factors = 4 * pi * areas ./ (perimeters.^2);
    
    % 汇总特征（均值和标准差）
    geo_features = [mean(areas), std(areas), ...
                    mean(shape_factors), std(shape_factors), ...
                    mean(eccentricities), std(eccentricities), ...
                    mean(solidities), mean(major_axes)];
end