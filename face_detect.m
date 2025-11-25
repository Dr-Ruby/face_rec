% //Yizuo Chen
function faces = detect_faces_image(imgFile, model)
% Detect faces using:
%   • RGB+HSV skin model (trained)
%   • Morphological cleanup
%   • Strong first-pass classification + template matching
%   • Automatic row-band estimation
%   • Rescue pass for missed faces
%   • Duplicate-merging (neck/logo removal)
% Final output: list of face bounding boxes + centers.

    I = imread(imgFile);
    if size(I,3) == 1
        I = repmat(I,[1 1 3]);   % handle grayscale input
    end
    Igray = im2double(rgb2gray(I));
    [H, W, ~] = size(I);

    % -------------------------------------------------------------
    % 1. Skin segmentation using trained RGB+HSV likelihood model
    % -------------------------------------------------------------
    % produces:
    %   (a) skinMask — thresholded bright skin-likelihood
    %   (b) logRimg  — visualization of log-likelihood
    [skinMask, logRimg] = apply_skin_model(I, model);
    debug_show(logRimg, 'DEBUG: logR image');

    % -------------------------------------------------------------
    % 1a. Morphology: merge patches, fill holes, remove noise
    % -------------------------------------------------------------
    % rely on training stats for minimal blob size
    minPix   = max(round(model.areaStats.min * 0.20), 60);

    skinMask = bwareaopen(skinMask, minPix);   % remove tiny blobs
    skinMask = imclose(skinMask, strel('disk', 6)); % merge forehead/cheek regions
    skinMask = imfill(skinMask, 'holes');      % remove holes from eyes/mouth
    skinMask = imopen(skinMask, strel('disk', 2));  % knock off thin noise
    skinMask = bwareaopen(skinMask, minPix);   % final size filter
    debug_show(skinMask, 'DEBUG: final skin mask');

    % -------------------------------------------------------------
    % 2. Connected component analysis
    % -------------------------------------------------------------
    % Extract geometric properties for all blobs.
    L     = bwlabel(skinMask);
    stats = regionprops(L, 'Area', 'BoundingBox', 'Centroid');

    % Debug display of all raw blobs
    if ~isempty(stats)
        imgBlobs = insertShape(I, 'Rectangle', cat(1,stats.BoundingBox), ...
                               'Color','cyan','LineWidth',2);
    else
        imgBlobs = I;
    end
    debug_show(imgBlobs, 'DEBUG: connected components');

    % Empty output if nothing detected
    faces = struct('BoundingBox', {}, 'Center', {}, 'Score', {});
    if isempty(stats)
        return;
    end

    % -------------------------------------------------------------
    % Load trained priors (area, vertical distribution, template)
    % -------------------------------------------------------------
    meanA   = model.areaStats.mean;
    minA    = 0.35 * meanA;   % conservative lower bound
    maxA    = 2.5  * meanA;   % reject very large artifacts

    yCenters = model.yPosHist.centers;
    yHist    = model.yPosHist.hist;

    tpl      = model.tpl;
    tplMask  = model.tplMask;
    tplSize  = model.tplSize;

    strongIdx = [];

    % -------------------------------------------------------------
    % 3. FIRST PASS — strong face candidates
    % Filters:
    %   • size, aspect ratio, fill ratio
    %   • vertical prior from training
    %   • skin intensity ratio
    %   • template correlation score
    % These survive as high-confidence faces.
    % -------------------------------------------------------------
    for i = 1:numel(stats)
        a  = stats(i).Area;
        bb = stats(i).BoundingBox;
        c  = stats(i).Centroid;

        w = bb(3);
        h = bb(4);
        ratio = h / w;

        % --- geometric constraints ---
        if a < minA || a > maxA
            continue;
        end
        if ratio < 0.8 || ratio > 2.0
            continue;
        end
        fillRatio = a / (w*h);
        if fillRatio < 0.30 || fillRatio > 0.90
            continue;
        end

        % --- vertical position prior ---
        yNorm = c(2) / H;
        if yNorm < 0.10 || yNorm > 0.70
            continue;
        end
        [~, idxY] = min(abs(yCenters - yNorm));
        if yHist(idxY) < 0.01
            continue;
        end

        % --- template correlation check ---
        pad = 0.25;
        x = bb(1); y = bb(2);
        x1 = max(1, floor(x - pad*w));
        y1 = max(1, floor(y - pad*h));
        x2 = min(W, ceil(x + w + pad*w));
        y2 = min(H, ceil(y + h + pad*h));

        % skin density in region
        patchLogR = logRimg(y1:y2, x1:x2);
        hardRatio = nnz(patchLogR > 0) / numel(patchLogR);
        if hardRatio < 0.07
            continue;
        end

        % correlation with template
        patch  = Igray(y1:y2, x1:x2);
        patchR = imresize(patch, tplSize);
        patchR(~tplMask) = 0;

        if var(patchR(:)) < 0.003
            continue;
        end

        tplVec   = tpl(:) - mean(tpl(:));
        patchVec = patchR(:) - mean(patchR(:));
        score = (tplVec' * patchVec) / (norm(tplVec)*norm(patchVec) + eps);

        if score < 0.40
            continue;
        end

        % --- accept strong candidate ---
        cx = x1 + (x2 - x1)/2;
        cy = y1 + (y2 - y1)/2;

        faces(end+1).BoundingBox = [x1 y1 (x2-x1+1) (y2-y1+1)];
        faces(end).Center        = [cx cy];
        faces(end).Score         = score;

        strongIdx(end+1) = i;
    end

    nStrong = numel(faces);

    % -------------------------------------------------------------
    % 4. Determine likely vertical row-band from strong detections
    % Used to rescue missed faces and eliminate table noise.
    % -------------------------------------------------------------
    if ~isempty(strongIdx)
        yStrongNorm = arrayfun(@(k) stats(k).Centroid(2) / H, strongIdx);
        margin = 0.05;
        rowBandMinNorm = max(0, min(yStrongNorm) - margin);
        rowBandMaxNorm = min(1, max(yStrongNorm) + margin);
    else
        % fallback if no strong detections
        rowBandMinNorm = 0.10;
        rowBandMaxNorm = 0.70;
    end

    % convert to pixel indices
    rowBandMinPix = max(1, floor(rowBandMinNorm * H));
    rowBandMaxPix = min(H, ceil(rowBandMaxNorm * H));

    % debug visualization of allowed rescue region
    bandMask = false(H,W);
    bandMask(rowBandMinPix:rowBandMaxPix, :) = true;
    debug_show(bandMask & skinMask, ...
        sprintf('DEBUG: face row band [%.2f, %.2f]', ...
        rowBandMinNorm, rowBandMaxNorm));

    % -------------------------------------------------------------
    % 4b. Adaptive LOWER bounds for rescued faces
    % Prevents tiny noise from entering second pass, but allows
    % large faces (edges of group) to be kept.
    % -------------------------------------------------------------
    if ~isempty(strongIdx)
        strongAreas   = [stats(strongIdx).Area];
        strongBB      = cat(1, stats(strongIdx).BoundingBox);
        strongHeights = strongBB(:,4);

        medA = median(strongAreas);
        medH = median(strongHeights);

        rescueMinA = 0.35 * medA;   % minimal area
        rescueMinH = 0.40 * medH;   % minimal height
    else
        rescueMinA = minA;
        rescueMinH = 0;
    end

    % -------------------------------------------------------------
    % 5. SECOND PASS — rescue plausible blobs inside row band
    % Used to recover faces missed in first pass.
    % -------------------------------------------------------------
    allIdx   = 1:numel(stats);
    extraIdx = setdiff(allIdx, strongIdx);

    for ii = extraIdx
        bb = stats(ii).BoundingBox;
        c  = stats(ii).Centroid;
        a  = stats(ii).Area;

        % must fall inside row band
        yNorm = c(2) / H;
        if yNorm < rowBandMinNorm || yNorm > rowBandMaxNorm
            continue;
        end

        % adaptive geometric filters
        w  = bb(3);
        h  = bb(4);
        ratio = h / w;
        fillRatio = a / (w*h);

        if a < rescueMinA, continue; end
        if h < rescueMinH, continue; end
        if ratio < 0.6 || ratio > 2.5, continue; end
        if fillRatio < 0.25 || fillRatio > 0.95, continue; end

        % accept rescued detection
        pad = 0.25;
        x = bb(1); y = bb(2);
        x1 = max(1, floor(x - pad*w));
        y1 = max(1, floor(y - pad*h));
        x2 = min(W, ceil(x + w + pad*w));
        y2 = min(H, ceil(y + h + pad*h));

        cx = x1 + (x2 - x1)/2;
        cy = y1 + (y2 - y1)/2;

        faces(end+1).BoundingBox = [x1 y1 (x2-x1+1) (y2-y1+1)];
        faces(end).Center        = [cx cy];
        faces(end).Score         = NaN;
    end

    % -------------------------------------------------------------
    % 6. Merge vertically stacked duplicates (logo + chin + head)
    % Removes doubled detections for same person.
    % -------------------------------------------------------------
    if numel(faces) > 1
        keep = true(1, numel(faces));

        for i = 1:numel(faces)
            if ~keep(i), continue; end
            b1 = faces(i).BoundingBox;
            c1 = faces(i).Center;
            w1 = b1(3); h1 = b1(4);

            for j = i+1:numel(faces)
                if ~keep(j), continue; end
                b2 = faces(j).BoundingBox;
                c2 = faces(j).Center;
                w2 = b2(3); h2 = b2(4);

                % horizontal similarity + vertical proximity
                wAvg = 0.5*(w1 + w2);
                dx   = abs(c1(1) - c2(1));
                dy   = abs(c1(2) - c2(2));

                if dx < 0.45*wAvg && dy < 1.4*max(h1,h2)
                    % pick stronger score as base
                    s1 = faces(i).Score; if isnan(s1), s1 = -Inf; end
                    s2 = faces(j).Score; if isnan(s2), s2 = -Inf; end

                    if s2 > s1
                        base = j; other = i; bbBase = b2; bbOther = b1;
                    else
                        base = i; other = j; bbBase = b1; bbOther = b2;
                    end

                    % merge bounding boxes
                    xMin = min(bbBase(1), bbOther(1));
                    yMin = min(bbBase(2), bbOther(2));
                    xMax = max(bbBase(1)+bbBase(3), bbOther(1)+bbOther(3));
                    yMax = max(bbBase(2)+bbBase(4), bbOther(2)+bbOther(4));

                    bbNew = [xMin, yMin, xMax-xMin, yMax-yMin];

                    faces(base).BoundingBox = bbNew;
                    faces(base).Center      = [bbNew(1)+bbNew(3)/2, ...
                                               bbNew(2)+bbNew(4)/2];

                    keep(other) = false;
                end
            end
        end

        faces = faces(keep);
    end

    % -------------------------------------------------------------
    % 7. Visualization of final detections
    % -------------------------------------------------------------
    outImg = I;
    for k = 1:numel(faces)
        outImg = insertShape(outImg,'Rectangle',faces(k).BoundingBox, ...
                             'Color','yellow','LineWidth',3);
    end

    debug_show(outImg, 'DEBUG: final detected faces');
end


%% ------------------------------------------------------------------------
% //Yizuo Chen
function [mask, logRimg] = apply_skin_model(I, model)
% Apply RGB+HSV log-likelihood model:
%   • compute skin probability using trained histograms
%   • convert to log-likelihood image
%   • apply bright-pixel threshold (~195–255)
%   • apply HSV gate to remove false positives

    I_rgb = im2uint8(I);
    [H, W, ~] = size(I_rgb);

    nbinsRGB = model.nbinsRGB;
    nbinsHSV = model.nbinsHSV;

    % ----------------- RGB flattening + binning -----------------
    R = double(reshape(I_rgb(:,:,1), [], 1));
    G = double(reshape(I_rgb(:,:,2), [], 1));
    B = double(reshape(I_rgb(:,:,3), [], 1));

    binR = floor(R / 256 * nbinsRGB) + 1;
    binG = floor(G / 256 * nbinsRGB) + 1;
    binB = floor(B / 256 * nbinsRGB) + 1;
    binR = min(max(binR,1), nbinsRGB);
    binG = min(max(binG,1), nbinsRGB);
    binB = min(max(binB,1), nbinsRGB);
    idxRGB = sub2ind([nbinsRGB nbinsRGB nbinsRGB], binR, binG, binB);

    % ----------------- HSV flattening + binning -----------------
    I_hsv = rgb2hsv(im2double(I_rgb));
    Hh = reshape(I_hsv(:,:,1), [], 1);
    Ss = reshape(I_hsv(:,:,2), [], 1);
    Vv = reshape(I_hsv(:,:,3), [], 1);

    binH = floor(Hh * nbinsHSV) + 1;
    binS = floor(Ss * nbinsHSV) + 1;
    binV = floor(Vv * nbinsHSV) + 1;
    binH = min(max(binH,1), nbinsHSV);
    binS = min(max(binS,1), nbinsHSV);
    binV = min(max(binV,1), nbinsHSV);

    idxHSV = sub2ind([nbinsHSV nbinsHSV nbinsHSV], binH, binS, binV);

    % ----------------- Combined log-likelihood -----------------
    scores  = model.logR_rgb(idxRGB) + model.logR_hsv(idxHSV) + model.alpha;
    logRimg = reshape(scores, H, W);

    % normalize to 0–255
    logRnorm = mat2gray(logRimg);
    logR8    = uint8(255 * logRnorm);

    % brightness thresholding (your measured skin limit)
    brightMask = logR8 >= 195;

    % HSV gate to reject bright non-skin (lights, shirts, monitors)
    Simg = reshape(Ss, H, W);
    Vimg = reshape(Vv, H, W);
    hsvGate = (Simg > 0.08) & (Simg < 0.95) & ...
              (Vimg > 0.05) & (Vimg < 0.98);

    mask = brightMask & hsvGate;
end


%% ------------------------------------------------------------------------
% //Yizuo Chen
function show_faces(imgFile, faces)
% Display detected faces with bounding boxes and center points.

    I = imread(imgFile);
    figure; imshow(I); hold on;

    for i = 1:numel(faces)
        bb = faces(i).BoundingBox;
        rectangle('Position', bb, 'EdgeColor', 'y', 'LineWidth', 2);
        plot(faces(i).Center(1), faces(i).Center(2), 'r+', ...
             'MarkerSize', 8, 'LineWidth', 2);
    end

    title(sprintf('Detected faces: %d', numel(faces)));
end


%% ------------------------------------------------------------------------
% //Yizuo Chen
function debug_show(img, titleText)
% Helper to show intermediate results with pause.

    figure('Name', titleText);
    imshow(img, []);
    title(titleText,'Interpreter','none');
    drawnow;
    uiwait(msgbox('OK to continue','DEBUG','modal'));
end

%%
load face_model.mat
faces = detect_faces_image("person4.jpg", model);
show_faces("person4.jpg", faces);
