% 使用训练好的KNN模型对新绝缘子图像进行分类
function className = classifyNewInsulator(imgPath, modelPath)
    % 加载模型
    load(modelPath, 'knnModel', 'classNames', 'mu', 'sigma');
    
    % 读取图像
    img = imread(imgPath);
    
    % 提取特征
    features = extractInsulatorFeatures(img);
    
    % 标准化特征
    features = (features - mu) ./ sigma;
    
    % 预测类别
    classIdx = predict(knnModel, features);
    
    % 返回类别名称
    className = classNames{classIdx};
    
    % 显示结果
    figure;
    imshow(img);
    title(sprintf('预测类别: %s', className));
end    