% Source: https://github.com/saikatbsk/bagOfFeatures
mainPath = dir('Assignment08_data_expanded/TrainingDataset');
mainPath(ismember({mainPath.name}, {'.', '..'})) = []; 
classNum = size(mainPath,1);

Descriptors = cell(classNum,1);

for i = 1:classNum
    datasetPath = fullfile(mainPath(i).folder, mainPath(i).name);
    disp(mainPath(i).name);
    descriptor = FeaturesExtractorSurf(datasetPath);
    Descriptors{i} = descriptor;
end

numCluster = 1000;
thresh = 10;
histograms = double(zeros(classNum, numCluster));

DesMat = cell2mat(Descriptors);
[idx,C] = kmeans(double(DesMat), numCluster, 'Distance', 'sqeuclidean');

for i = 1:classNum
    descriptor = Descriptors{i};
    [IDX, D] = knnsearch(double(C), double(descriptor), 'Distance','seuclidean');
    for j=1:size(IDX,1)
        if D(j) <= thresh
            histograms(i, IDX(j)) = histograms(i, IDX(j)) + 1;
        end
    end
end
for i = 1:classNum
    sum_bin = sum(histograms(i,:));
    histograms(i,:) = double(histograms(i,:))/sum_bin;
end

confusionMat = zeros(25, 25);
keySet = {'006', '007', '011', '012', '022', '024', '025', '026', '028', '031', '037', '045', '051', '054', '063', '072', '093', '102', '129', '145', '159', '171', '178', '180', '251'};
valueSet = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
Map = containers.Map(keySet,valueSet);

filePath = 'Assignment08_data_expanded/TestDataset';
imagesPath = dir(filePath);
imagesPath(ismember({imagesPath.name}, {'.', '..'})) = [];
ImgNum = size(imagesPath,1);
for j=1:ImgNum
    baseName = imagesPath(j).name;
    fullName = fullfile(imagesPath(j).folder, baseName);
    I = imread(fullName);
    if size(I, 3) > 1
        I = rgb2gray(I);
    end
    points = detectSURFFeatures(I);
    testFeatures = extractFeatures(I, points, "Upright", true);
    [IDX, D] = knnsearch(double(C), double(testFeatures), 'Distance','seuclidean');
    testHist = double(zeros(1, numCluster));
    for k=1:size(IDX,1)
        if D(k) <= thresh
            testHist(1, IDX(k)) = testHist(1, IDX(k)) + 1;
        end
    end
    testHist  = testHist / sum(testHist);
    IDX = knnsearch(double(histograms), double(testHist));
    fprintf('Image Name: %s, Class Name: %s.\n', baseName, mainPath(IDX).name);
    x = Map(baseName(1:3));
    y = Map(mainPath(IDX).name(1:3));
    confusionMat(x, y) = confusionMat(x, y)+1;
end


function descriptor = FeaturesExtractorSurf(datasetPath)
    imagesPath = dir(datasetPath);
    imagesPath(ismember({imagesPath.name}, {'.', '..'})) = []; 
    imgNum = size(imagesPath,1);
    for i = 1:imgNum
        I = fullfile(imagesPath(i).folder, imagesPath(i).name);
        I = imread(I);
        if size(I, 3) > 1
            I = rgb2gray(I);
        end
        points = detectSURFFeatures(I);
        features = extractFeatures(I, points, "Upright", true);
        if i == 1
            descriptor = features;
        else
            descriptor = [descriptor; features];
        end
    end
end

    
