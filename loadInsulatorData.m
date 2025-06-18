% 加载绝缘子数据并提取特征
function [features, labels] = loadInsulatorData(dataPath, classNames)
    % 初始化
    features = [];
    labels = [];
    
    % 对每个类别
    for i = 1:length(classNames)
        className = classNames{i};
        classPath = fullfile(dataPath, className);
        
        % 获取所有图像文件
        imageFiles = dir(fullfile(classPath, '*.jpg'));
        if isempty(imageFiles)
            imageFiles = dir(fullfile(classPath, '*.png'));
        end
        
        % 处理每个图像
        for j = 1:length(imageFiles)
            % 读取图像
            imgPath = fullfile(classPath, imageFiles(j).name);
            img = imread(imgPath);
            
            % 提取特征
            imgFeatures = extractInsulatorFeatures(img);
            
            % 添加到特征矩阵和标签向量
            features = [features; imgFeatures];
            labels = [labels; i];
            
            % 显示进度
            fprintf('已处理 %s 类别的 %d/%d 图像\n', className, j, length(imageFiles));
        end
    end
end    