% 绝缘子憎水性KNN识别算法 - 主函数
clear all; close all; clc;

% 设置参数
k = 5; % KNN的K值
dataPath = 'test/'; % 数据路径
classNames = {'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7'}; % 类别名称

% 加载数据
fprintf('正在加载数据...\n');
[features, labels] = loadInsulatorData(dataPath, classNames);

% 划分训练集和测试集 (70%训练，30%测试)
cv = cvpartition(size(features,1),'HoldOut',0.3);
idxTrain = training(cv);
idxTest = test(cv);

% 提取训练集和测试集
X_train = features(idxTrain,:);
y_train = labels(idxTrain);
X_test = features(idxTest,:);
y_test = labels(idxTest);

% 训练KNN模型
fprintf('正在训练KNN模型 (k=%d)...\n', k);
knnModel = fitcknn(X_train, y_train, 'NumNeighbors', k);

% 预测测试集
fprintf('正在预测测试集...\n');
y_pred = predict(knnModel, X_test);

% 评估模型
fprintf('正在评估模型性能...\n');
accuracy = mean(y_pred == y_test) * 100;
fprintf('模型准确率: %.2f%%\n', accuracy);

% 显示混淆矩阵
figure;
cm = confusionchart(y_test, y_pred);
cm.Title = '绝缘子憎水性分类混淆矩阵';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

% 保存模型
save('insulator_knn_model.mat', 'knnModel', 'classNames');
fprintf('模型已保存为 insulator_knn_model.mat\n');    