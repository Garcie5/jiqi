function color_features = extract_color_features(img)
    % 提取颜色特征
    % 输入: img - 图像
    % 输出: color_features - 提取的颜色特征向量
    
    % 转换到HSV颜色空间
    img_hsv = rgb2hsv(img);
    
    % 计算各通道的直方图
    nbins = 8;  % 每个通道的直方图bin数量
    hist_r = imhist(img(:,:,1), nbins);
    hist_g = imhist(img(:,:,2), nbins);
    hist_b = imhist(img(:,:,3), nbins);
    
    % 归一化直方图
    hist_r = hist_r / sum(hist_r);
    hist_g = hist_g / sum(hist_g);
    hist_b = hist_b / sum(hist_b);
    
    % 连接直方图作为特征向量
    color_features = [hist_r; hist_g; hist_b];
end