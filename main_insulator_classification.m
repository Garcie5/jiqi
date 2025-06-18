% 主程序：绝缘子憎水性KNN分类
clear all; close all; clc;

%% 数据路径配置
dataDir = 'D:\水滴等级\test'; % 根据你的路径修改
classNames = {'HC1', 'HC2', 'HC3', 'HC4', 'HC5', 'HC6', 'HC7'};
numClasses = length(classNames);

%% 数据加载与特征提取
% 初始化特征和标签数组
features = [];
labels = [];
expectedFeatureSize = 218; % 明确指定期望的特征维度

% 遍历每个类别文件夹
for i = 1:numClasses
    classDir = fullfile(dataDir, classNames{i});
    
    % 检查文件夹是否存在
    if ~exist(classDir, 'dir')
        error(sprintf('类别文件夹 %s 不存在！请检查路径: %s', classNames{i}, classDir));
    end
    
    % 获取当前类别下的所有图像文件
    imageFiles = dir(fullfile(classDir, '*.jpg'));
    if isempty(imageFiles)
        imageFiles = dir(fullfile(classDir, '*.png'));
    end
    
    % 检查是否找到图像
    if isempty(imageFiles)
        error(sprintf('在类别 %s 中未找到图像文件！路径: %s', classNames{i}, classDir));
    end
    
    % 输出当前处理的类别和图像数量
    fprintf('处理类别 %s，找到 %d 张图像\n', classNames{i}, length(imageFiles));
    
    % 遍历当前类别下的所有图像
    for j = 1:length(imageFiles)
        % 读取图像
        imgPath = fullfile(classDir, imageFiles(j).name);
        img = imread(imgPath);
        
        % 提取特征
        imgFeatures = extractInsulatorFeatures(img);
        
        % 确保特征维度正确
        if ~isempty(imgFeatures) && size(imgFeatures, 2) == expectedFeatureSize
            % 特征维度匹配，添加到矩阵
            features = [features; imgFeatures];
            labels = [labels; i];
        else
            % 特征维度不匹配，使用默认特征
            warning(sprintf('图像 %s 的特征维度不匹配！期望 %d 列，实际 %d 列。使用默认特征。', ...
                imgPath, expectedFeatureSize, size(imgFeatures, 2)));
            defaultFeatures = zeros(1, expectedFeatureSize);
            features = [features; defaultFeatures];
            labels = [labels; i];
        end
    end
end

% 检查是否有有效的特征数据
if isempty(features)
    error('未能从任何图像中提取有效特征！请检查图像格式和特征提取函数。');
else
    fprintf('成功提取 %d 个样本的特征，每个样本有 %d 个特征\n', size(features, 1), size(features, 2));
end

%% 模型训练与评估
% 创建交叉验证分区
cv = cvpartition(size(features,1), 'HoldOut', 0.3);

% 获取训练集和测试集索引
idxTrain = training(cv);
idxTest = test(cv);

% 划分训练集和测试集
X_train = features(idxTrain,:);
y_train = labels(idxTrain);
X_test = features(idxTest,:);
y_test = labels(idxTest);

% 数据标准化
mu = mean(X_train);
sigma = std(X_train);
X_train = (X_train - repmat(mu, size(X_train,1), 1)) ./ repmat(sigma, size(X_train,1), 1);
X_test = (X_test - repmat(mu, size(X_test,1), 1)) ./ repmat(sigma, size(X_test,1), 1);

% 训练KNN模型
k = 5; % K值，可以根据需要调整
knnModel = fitcknn(X_train, y_train, 'NumNeighbors', k);

% 在测试集上进行预测
y_pred = predict(knnModel, X_test);

% 评估模型性能
accuracy = sum(y_pred == y_test) / length(y_test);
fprintf('分类准确率: %.2f%%\n', accuracy*100);

% 绘制混淆矩阵
figure;
cm = confusionmat(y_test, y_pred);
imagesc(cm);
colorbar;
title('绝缘子憎水性分类混淆矩阵');
xlabel('预测类别');
ylabel('真实类别');

% 添加数值标签
[nrows, ncols] = size(cm);
for i = 1:nrows
    for j = 1:ncols
        text(j, i, num2str(cm(i,j)), 'HorizontalAlignment', 'center', 'Color', 'white');
    end
end

% 设置坐标轴刻度和标签
set(gca, 'XTick', 1:ncols, 'XTickLabel', classNames);
set(gca, 'YTick', 1:nrows, 'YTickLabel', classNames);

% 保存模型和标准化参数
save('insulator_knn_model.mat', 'knnModel', 'classNames', 'mu', 'sigma');
fprintf('模型已保存为 insulator_knn_model.mat\n');    