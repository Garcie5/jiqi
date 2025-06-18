figure;
cm = confusionchart(testLabels, predictions);
cm.Title = '绝缘子水滴等级分类混淆矩阵';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';