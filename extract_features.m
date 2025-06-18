function [geo_features, texture_features, color_features] = extract_features(img_processed, img_original)
    % 几何特征
    img_binary = imbinarize(img_processed);
    geo_features = extract_geometric_features(img_binary);
    
    % 纹理特征
    texture_features = extract_texture_features(img_processed);
    
    % 颜色特征
    img_resized = imresize(img_original, [224, 224]);
    color_features = extract_color_features(img_resized);
end