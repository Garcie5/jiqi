%% 绝缘子水滴等级KNN分类 
clc; clear; close all;
warning('off', 'images:imadjust:lowHighPercentiles');

%% 1. 路径设置与数据检查
basePath = 'D:\水滴等级\';
trainPath = fullfile(basePath, 'train');
assert(isfolder(trainPath), '训练路径不存在: %s', trainPath);

%% 2. 自动配置HOG参数
sampleImgPath = dir(fullfile(trainPath, '**/*.jpg'));
sampleImg = preprocessImage(imread(fullfile(sampleImgPath(1).folder, sampleImgPath(1).name)));
[hogFeatures, ~] = extractHOGFeatures(sampleImg, 'CellSize', [8 8]);
hogFeatureSize = length(hogFeatures);
fprintf('HOG特征配置:\n 特征维度:%d\n 单元格大小:[8 8]\n', hogFeatureSize);

%% 3. 数据加载
imdsTrain = imageDatastore(trainPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', {'.jpg', '.png', '.bmp'});
assert(~isempty(imdsTrain.Files), '训练集未找到图像文件！');

%% 4. 特征提取
trainFeatures = zeros(numel(imdsTrain.Files), hogFeatureSize, 'single');
parfor i = 1:numel(imdsTrain.Files)
    img = preprocessImage(readimage(imdsTrain, i));
    trainFeatures(i, :) = extractHOGFeatures(img, 'CellSize', [8 8]);
end
trainLabels = imdsTrain.Labels;
classNames = categories(trainLabels);

%% 5. KNN模型训练与评估
kValues = 1:2:10;
cvAcc = zeros(length(kValues), 1);
bestConfMat = zeros(numel(classNames)); % 存储最佳混淆矩阵

% 交叉验证框架
cvp = cvpartition(trainLabels, 'KFold', 5, 'Stratify', true);

for i = 1:length(kValues)
    tempConfMat = zeros(numel(classNames)); % 临时存储当前K值的混淆矩阵
    
    for j = 1:cvp.NumTestSets
        trainIdx = training(cvp,j);
        testIdx = test(cvp,j);
        
        model = fitcknn(trainFeatures(trainIdx,:), trainLabels(trainIdx), ...
            'NumNeighbors', kValues(i), ...
            'Standardize', true, ...
            'Distance', 'cosine');
        
        pred = predict(model, trainFeatures(testIdx,:));
        cvAcc(i) = cvAcc(i) + sum(pred == trainLabels(testIdx));
        tempConfMat = tempConfMat + confusionmat(trainLabels(testIdx), pred, 'Order', classNames);
    end
    
    cvAcc(i) = cvAcc(i) / numel(trainLabels) * 100;
    tempConfMat = tempConfMat ./ sum(tempConfMat,2); % 归一化
    
    if i == 1 || cvAcc(i) > max(cvAcc(1:i-1))
        bestConfMat = tempConfMat; % 更新最佳混淆矩阵
    end
    
    fprintf('K=%d 交叉验证准确率: %.2f%%\n', kValues(i), cvAcc(i));
end

[bestAcc, bestIdx] = max(cvAcc);
optimalK = kValues(bestIdx);
fprintf('\n最优K值: %d (准确率 %.2f%%)\n', optimalK, bestAcc);

% 可视化混淆矩阵
figure('Name', '最佳K值混淆矩阵', 'NumberTitle', 'off');
h = heatmap(classNames, classNames, bestConfMat*100, ...
    'Colormap', parula, ...
    'ColorLimits', [0 100], ...
    'ColorbarVisible', 'on');
h.Title = sprintf('K=%d 归一化混淆矩阵 (准确率: %.1f%%)', optimalK, bestAcc);
h.XLabel = '预测类别';
h.YLabel = '真实类别';

% 训练最终模型
finalModel = fitcknn(trainFeatures, trainLabels, ...
    'NumNeighbors', optimalK, ...
    'Standardize', true, ...
    'Distance', 'cosine');

%% 6. 单图预测功能 (保持不变)
while true
    [file, path] = uigetfile({'*.jpg;*.png;*.bmp', 'Image Files'}, '选择图像 (取消退出)');
    if isequal(file, 0), break; end
    
    try
        img = preprocessImage(imread(fullfile(path, file)));
        imgFeatures = extractHOGFeatures(img, 'CellSize', [8 8]);
        predictedLabel = predict(finalModel, imgFeatures);
        
        % 显示结果
        fig = figure('Name', '预测结果', 'Position', [200 200 900 400]);
        subplot(1,3,1);
        imshow(img);
        title(sprintf('输入图像: %s', file), 'Interpreter', 'none');
        
        subplot(1,3,2);
        text(0.1, 0.5, sprintf('预测等级:\n%s', char(predictedLabel)), ...
            'FontSize', 16, 'Color', [0 0.5 0]);
        axis off;
        
        % 在预测界面添加混淆矩阵提示
        subplot(1,3,3);
        imshow(imread('confusion_matrix_placeholder.jpg')); % 替换为实际矩阵图像或文本说明
        title('模型混淆矩阵参考');
        axis off;
        
    catch ME
        errordlg(sprintf('处理失败: %s', ME.message), '错误');
    end
end

%% 预处理函数
function img = preprocessImage(img)
    if size(img, 3) == 3, img = rgb2gray(img); end
    img = imresize(img, [128 128]);
    img = imadjust(img);
    img = imgaussfilt(img, 1);
end