function features = extract_texture_features(img)
    % 确保图像是灰度图
    if size(img, 3) > 1
        img = rgb2gray(img);
    end
    
    % 使用灰度共生矩阵(GLCM)提取纹理特征
    glcm = graycomatrix(img, 'NumLevels', 8, 'Offset', [0 1; 1 0; 1 1; -1 1]);
    stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    
    % 计算每个特征的均值作为最终特征向量
    features = [mean(stats.Contrast), mean(stats.Correlation), ...
                mean(stats.Energy), mean(stats.Homogeneity)];
end