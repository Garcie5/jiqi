% 从绝缘子图像中提取特征
function features = extractInsulatorFeatures(img)
    try
        % 转换为灰度图
        if size(img, 3) == 3
            grayImg = rgb2gray(img);
        else
            grayImg = img;
        end
        
        % 图像预处理
        img = imresize(grayImg, [256, 256]); % 调整大小
        img = imgaussfilt(img, 1); % 高斯滤波
        
        % 提取颜色特征 (RGB均值)
        colorFeatures = zeros(1, 3);
        if size(img, 3) == 3
            colorFeatures = [mean(mean(img(:,:,1))), mean(mean(img(:,:,2))), mean(mean(img(:,:,3)))];
        else
            colorFeatures = [mean(mean(img)), mean(mean(img)), mean(mean(img))];
        end
        
        % 提取纹理特征 (使用GLCM)
        textureFeatures = zeros(1, 4);
        try
            glcm = graycomatrix(img, 'Offset', [0 1; 1 0; 1 1; 1 -1], 'NumLevels', 8, 'Symmetric', true);
            stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
            textureFeatures = [mean([stats.Contrast]), mean([stats.Correlation]), mean([stats.Energy]), mean([stats.Homogeneity])];
        catch
            warning('GLCM纹理特征提取失败，使用默认值');
        end
        
        % 提取形状特征
        shapeFeatures = zeros(1, 13); % 7个regionprops属性 + 7个Hu矩 + 1个面积
        try
            % 改进的二值化方法
            bwImg = imbinarize(img, 'adaptive'); % 使用自适应阈值
            
            % 检查二值图像是否有效
            if sum(bwImg(:)) > 100 % 确保有足够的前景像素
                % 填充孔洞
                bwImg = imfill(bwImg, 'holes');
                
                % 移除小对象
                bwImg = bwareaopen(bwImg, 50);
                
                % 提取区域属性
                regionProps = regionprops(bwImg, 'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', ...
                                         'Eccentricity', 'Solidity', 'PixelIdxList');
                
                if ~isempty(regionProps)
                    % 计算Hu矩
                    pixelList = regionProps.PixelIdxList;
                    if ~isempty(pixelList)
                        % 计算中心矩
                        [rows, cols] = ind2sub(size(bwImg), pixelList);
                        m00 = regionProps.Area;
                        m10 = sum(rows);
                        m01 = sum(cols);
                        xc = m10 / m00;  % 质心x坐标
                        yc = m01 / m00;  % 质心y坐标
                        
                        % 计算归一化中心矩
                        u20 = sum((rows - xc).^2) / m00^2;
                        u02 = sum((cols - yc).^2) / m00^2;
                        u11 = sum((rows - xc) .* (cols - yc)) / m00^2;
                        u30 = sum((rows - xc).^3) / m00^2.5;
                        u03 = sum((cols - yc).^3) / m00^2.5;
                        u21 = sum((rows - xc).^2 .* (cols - yc)) / m00^2.5;
                        u12 = sum((rows - xc) .* (cols - yc).^2) / m00^2.5;
                        
                        % 计算Hu矩
                        h1 = u20 + u02;
                        h2 = (u20 - u02)^2 + 4*u11^2;
                        h3 = (u30 - 3*u12)^2 + (3*u21 - u03)^2;
                        h4 = (u30 + u12)^2 + (u21 + u03)^2;
                        h5 = (u30 - 3*u12)*(u30 + u12)*((u30 + u12)^2 - 3*(u21 + u03)^2) + ...
                             (3*u21 - u03)*(u21 + u03)*(3*(u30 + u12)^2 - (u21 + u03)^2);
                        h6 = (u20 - u02)*((u30 + u12)^2 - (u21 + u03)^2) + ...
                             4*u11*(u30 + u12)*(u21 + u03);
                        h7 = (3*u21 - u03)*(u30 + u12)*((u30 + u12)^2 - 3*(u21 + u03)^2) - ...
                             (u30 - 3*u12)*(u21 + u03)*(3*(u30 + u12)^2 - (u21 + u03)^2);
                        
                        huMoments = [h1, h2, h3, h4, h5, h6, h7];
                    else
                        huMoments = zeros(1, 7);
                    end
                    
                    % 组合形状特征
                    shapeFeatures = [regionProps.Area, regionProps.Perimeter, regionProps.MajorAxisLength, ...
                                    regionProps.MinorAxisLength, regionProps.Eccentricity, regionProps.Solidity, huMoments];
                end
            end
        catch e
            warning(sprintf('形状特征提取失败: %s', e.message));
        end
        
        % 组合所有特征
        features = [colorFeatures, textureFeatures, shapeFeatures];
        
        % 确保特征维度正确
        if size(features, 2) ~= 218
            warning(sprintf('特征维度异常！期望218列，实际%d列。调整维度。', size(features, 2)));
            features = [features, zeros(1, 218 - size(features, 2))]; % 填充到218列
        end
        
    catch e
        % 如果发生严重错误，返回默认特征向量
        warning(sprintf('特征提取严重错误: %s', e.message));
        features = zeros(1, 218); % 默认特征向量
    end
end    