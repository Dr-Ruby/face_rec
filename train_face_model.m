% //Yizuo Chen
function model = train_face_models()
% Train RGB/HSV skin likelihood model + face template
% using Training_1.jpg..Training_11.jpg and ref masks.
% This model is later used for:
%   - Skin detection (apply_skin_model)
%   - Template matching (strong detections)
%   - Basic size and position statistics

    % --------------------------------------------------------------
    % Training parameters
    % --------------------------------------------------------------
    numTrain   = 11;         % # training images
    nbinsRGB   = 32;         % RGB histogram bin count
    nbinsHSV   = 32;         % HSV histogram bin count
    sigmaHist  = 1.0;        % Gaussian smoothing for histograms
    tplH       = 80;         % template height (pixels)
    tplW       = 60;         % template width
    nBinsYpos  = 20;         % vertical-position histogram bins
    alphaGrid  = -2:0.2:4;   % candidate alpha thresholds

    % --------------------------------------------------------------
    % Histogram accumulators (RGB + HSV)
    % --------------------------------------------------------------
    H_face_rgb = zeros(nbinsRGB, nbinsRGB, nbinsRGB);
    H_bg_rgb   = zeros(nbinsRGB, nbinsRGB, nbinsRGB);
    H_face_hsv = zeros(nbinsHSV, nbinsHSV, nbinsHSV);
    H_bg_hsv   = zeros(nbinsHSV, nbinsHSV, nbinsHSV);

    % Accumulated geometry + template data
    areaList = [];
    yList    = [];
    sumTemplate = zeros(tplH, tplW);
    faceCount   = 0;

    % Elliptical mask to define valid pixels in template
    [xx, yy] = meshgrid(linspace(-1,1,tplW), linspace(-1,1,tplH));
    ellMask = (xx.^2/0.9^2 + yy.^2/1.0^2) <= 1;

    % Store pixel indices for alpha tuning
    allIdxRGB = {};
    allIdxHSV = {};
    allGT     = {};

    fprintf('Training on %d images (RGB + HSV)...\n', numTrain);

    % --------------------------------------------------------------
    % Loop through all training pairs (image + GT mask)
    % --------------------------------------------------------------
    for k = 1:numTrain
        imgName = sprintf('Training_%d.jpg', k);
        refName = sprintf('ref%d.png', k);

        I  = im2uint8(imread(imgName));
        GT = imread(refName);

        if size(GT,3) > 1
            GT = rgb2gray(GT);
        end
        GT = GT > 0;   % ground-truth mask

        % ----------------------------------------------------------
        % Build RGB/HSV histograms for face vs. background
        % ----------------------------------------------------------
        [Hf_rgb, Hb_rgb, Hf_hsv, Hb_hsv, idxRGB, idxHSV, pixelGT] = ...
            build_color_histograms(I, GT, nbinsRGB, nbinsHSV);

        % Accumulate histograms
        H_face_rgb = H_face_rgb + Hf_rgb;
        H_bg_rgb   = H_bg_rgb   + Hb_rgb;
        H_face_hsv = H_face_hsv + Hf_hsv;
        H_bg_hsv   = H_bg_hsv   + Hb_hsv;

        % Store pixel mappings for alpha optimization
        allIdxRGB{end+1} = idxRGB;
        allIdxHSV{end+1} = idxHSV;
        allGT{end+1}     = pixelGT;

        % ----------------------------------------------------------
        % Extract face components for:
        %  - size statistics
        %  - vertical position stats
        %  - template averaging
        % ----------------------------------------------------------
        L = bwlabel(GT);
        stats = regionprops(L, 'Area', 'BoundingBox', 'Centroid');

        for i = 1:numel(stats)
            a  = stats(i).Area;
            bb = stats(i).BoundingBox;
            c  = stats(i).Centroid;

            % Collect global statistics
            areaList(end+1) = a;
            yList(end+1)    = c(2) / size(GT,1);

            % Crop expanded region around face (includes chin/hair)
            pad = 0.3;
            x = bb(1); y = bb(2); w = bb(3); h = bb(4);
            x1 = max(1, floor(x - pad*w));
            y1 = max(1, floor(y - pad*h));
            x2 = min(size(I,2), ceil(x + w + pad*w));
            y2 = min(size(I,1), ceil(y + h + pad*h));
            rect = [x1 y1 x2-x1+1 y2-y1+1];

            patchRGB  = imcrop(I, rect);
            patchGray = rgb2gray(patchRGB);
            patchGray = im2double(imresize(patchGray, [tplH tplW]));

            % Apply elliptical mask around face region
            patchGray(~ellMask) = 0;

            % Accumulate template
            sumTemplate = sumTemplate + patchGray;
            faceCount   = faceCount + 1;
        end
    end

    if faceCount == 0
        error('No faces found in training masks.');
    end

    % --------------------------------------------------------------
    % Smooth histograms + convert to probability densities
    % --------------------------------------------------------------
    fprintf('Smoothing 3D histograms...\n');
    G_rgb = gaussian3d_kernel(sigmaHist);
    G_hsv = gaussian3d_kernel(sigmaHist);

    H_face_rgb_s = convn(H_face_rgb, G_rgb, 'same');
    H_bg_rgb_s   = convn(H_bg_rgb,   G_rgb, 'same');
    H_face_hsv_s = convn(H_face_hsv, G_hsv, 'same');
    H_bg_hsv_s   = convn(H_bg_hsv,   G_hsv, 'same');

    % Normalize to proper PDFs, avoid zero probabilities
    P_face_rgb = H_face_rgb_s + 1;
    P_bg_rgb   = H_bg_rgb_s   + 1;
    P_face_hsv = H_face_hsv_s + 1;
    P_bg_hsv   = H_bg_hsv_s   + 1;

    P_face_rgb = P_face_rgb / sum(P_face_rgb(:));
    P_bg_rgb   = P_bg_rgb   / sum(P_bg_rgb(:));
    P_face_hsv = P_face_hsv / sum(P_face_hsv(:));
    P_bg_hsv   = P_bg_hsv   / sum(P_bg_hsv(:));

    % Log-likelihood ratios for classification
    logR_rgb = log(P_face_rgb) - log(P_bg_rgb);
    logR_hsv = log(P_face_hsv) - log(P_bg_hsv);

    logR_rgb_flat = logR_rgb(:);
    logR_hsv_flat = logR_hsv(:);

    % --------------------------------------------------------------
    % Tune alpha parameter for optimal skin classification
    % --------------------------------------------------------------
    fprintf('Tuning alpha (RGB + HSV)...\n');
    bestAlpha = alphaGrid(1);
    bestErr   = inf;
    totalPix  = 0;

    for a = alphaGrid
        errCount = 0;
        pixCount = 0;

        for k = 1:numTrain
            idxR = allIdxRGB{k};
            idxH = allIdxHSV{k};
            gt   = allGT{k};

            % Combined RGB + HSV discriminator
            scores = logR_rgb_flat(idxR) + logR_hsv_flat(idxH) + a;
            pred   = scores > 0;

            errCount = errCount + sum(pred ~= gt);
            pixCount = pixCount + numel(gt);
        end

        errRate = errCount / pixCount;
        if errRate < bestErr
            bestErr   = errRate;
            bestAlpha = a;
        end
        totalPix = pixCount;
    end

    fprintf('Best alpha = %.3f, training error = %.4f\n', ...
            bestAlpha, bestErr);

    % --------------------------------------------------------------
    % Area + vertical statistics for prior information
    % --------------------------------------------------------------
    meanArea = mean(areaList);
    stdArea  = std(areaList);
    minArea  = min(areaList);

    [yHist, yEdges] = histcounts(yList, nBinsYpos, ...
                                 'BinLimits',[0 1], ...
                                 'Normalization','probability');
    yCenters = (yEdges(1:end-1) + yEdges(2:end)) / 2;

    % --------------------------------------------------------------
    % Build average face template (normalized)
    % --------------------------------------------------------------
    avgTpl = sumTemplate / faceCount;
    avgTpl = avgTpl - min(avgTpl(:));
    if max(avgTpl(:)) > 0
        avgTpl = avgTpl / max(avgTpl(:));
    end
    avgTpl(~ellMask) = 0;

    % --------------------------------------------------------------
    % Pack all learned components into model struct
    % --------------------------------------------------------------
    model = struct();
    model.nbinsRGB   = nbinsRGB;
    model.nbinsHSV   = nbinsHSV;

    model.P_face_rgb = P_face_rgb;
    model.P_bg_rgb   = P_bg_rgb;
    model.P_face_hsv = P_face_hsv;
    model.P_bg_hsv   = P_bg_hsv;

    model.logR_rgb   = logR_rgb_flat;
    model.logR_hsv   = logR_hsv_flat;
    model.alpha      = bestAlpha;

    model.tpl        = avgTpl;
    model.tplMask    = ellMask;
    model.tplSize    = [tplH tplW];

    model.areaStats  = struct('mean', meanArea, 'std', stdArea, 'min', minArea);
    model.yPosHist   = struct('centers', yCenters, 'hist', yHist);

    model.trainErr   = bestErr;
    model.totalPixels = totalPix;

    fprintf('Training complete. Faces accumulated for template: %d\n', faceCount);
end

% -------------------------------------------------------------------------
% Helper: build RGB/HSV histograms from a single training image
% -------------------------------------------------------------------------
function [H_face_rgb, H_bg_rgb, H_face_hsv, H_bg_hsv, idxRGB, idxHSV, pixelGT] = ...
         build_color_histograms(I, GT, nbinsRGB, nbinsHSV)

    I_rgb = im2uint8(I);
    I_hsv = rgb2hsv(im2double(I_rgb));
    [H, W, ~] = size(I_rgb);

    % Flatten image channels
    R = double(reshape(I_rgb(:,:,1), [], 1));
    G = double(reshape(I_rgb(:,:,2), [], 1));
    B = double(reshape(I_rgb(:,:,3), [], 1));

    Hh = reshape(I_hsv(:,:,1), [], 1);
    Ss = reshape(I_hsv(:,:,2), [], 1);
    Vv = reshape(I_hsv(:,:,3), [], 1);

    gt = reshape(GT, [], 1) > 0;

    % --------------------- RGB bin computation ---------------------
    binR = floor(R / 256 * nbinsRGB) + 1;
    binG = floor(G / 256 * nbinsRGB) + 1;
    binB = floor(B / 256 * nbinsRGB) + 1;

    binR = min(max(binR,1), nbinsRGB);
    binG = min(max(binG,1), nbinsRGB);
    binB = min(max(binB,1), nbinsRGB);

    idxRGB = sub2ind([nbinsRGB nbinsRGB nbinsRGB], binR, binG, binB);

    % --------------------- HSV bin computation ---------------------
    binH = floor(Hh * nbinsHSV) + 1;
    binS = floor(Ss * nbinsHSV) + 1;
    binV = floor(Vv * nbinsHSV) + 1;

    binH = min(max(binH,1), nbinsHSV);
    binS = min(max(binS,1), nbinsHSV);
    binV = min(max(binV,1), nbinsHSV);

    idxHSV = sub2ind([nbinsHSV nbinsHSV nbinsHSV], binH, binS, binV);

    % --------------------- Build RGB histograms ---------------------
    idxFaceRGB = idxRGB(gt);
    H_face_rgb = accumarray(idxFaceRGB, 1, [nbinsRGB^3 1]);
    H_face_rgb = reshape(H_face_rgb, [nbinsRGB nbinsRGB nbinsRGB]);

    idxBgRGB = idxRGB(~gt);
    H_bg_rgb = accumarray(idxBgRGB, 1, [nbinsRGB^3 1]);
    H_bg_rgb = reshape(H_bg_rgb, [nbinsRGB nbinsRGB nbinsRGB]);

    % --------------------- Build HSV histograms ---------------------
    idxFaceHSV = idxHSV(gt);
    H_face_hsv = accumarray(idxFaceHSV, 1, [nbinsHSV^3 1]);
    H_face_hsv = reshape(H_face_hsv, [nbinsHSV nbinsHSV nbinsHSV]);

    idxBgHSV = idxHSV(~gt);
    H_bg_hsv = accumarray(idxBgHSV, 1, [nbinsHSV^3 1]);
    H_bg_hsv = reshape(H_bg_hsv, [nbinsHSV nbinsHSV nbinsHSV]);

    % Ground truth labels for alpha tuning
    pixelGT = gt;
end

% -------------------------------------------------------------------------
% Helper: build 3D Gaussian kernel
% -------------------------------------------------------------------------
function G = gaussian3d_kernel(sigma)
    r = ceil(3*sigma);
    [x, y, z] = ndgrid(-r:r, -r:r, -r:r);
    G = exp(-(x.^2 + y.^2 + z.^2) / (2*sigma^2));
    G = G / sum(G(:));
end

% Train and save
model = train_face_models();
save("face_model.mat","model");
