function img_processed = preprocess_image(img)
    % 灰度转换
    if size(img, 3) == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end
    
    % 对比度增强
    img_eq = adapthisteq(img_gray, 'ClipLimit', 0.03);
    
    % 降噪
    img_smooth = imgaussfilt(img_eq, 1);
    
    % 二值化分割
    level = graythresh(img_smooth);
    img_binary = imbinarize(img_smooth, level);
    
    % 形态学操作
    se = strel('disk', 3);
    img_morph = imclose(imopen(img_binary, se), se);
    
    % 提取最大区域作为ROI
    [B, L] = bwboundaries(img_morph);
    stats = regionprops(L, 'Area', 'BoundingBox');
    if ~isempty(stats)
        [~, idx] = max([stats.Area]);
        img_processed = imcrop(img_smooth, stats(idx).BoundingBox);
    else
        img_processed = img_smooth;
    end
    
    % 统一尺寸
    img_processed = imresize(img_processed, [224, 224]);
end