% //Yizuo Chen
% Clean ref8.png so that all non-white pixels become pure black.
% Output: ref8_clean.png (binary mask)

function clean_ref8()

    % --- Load image ---
    I = imread('ref4.png');

    % Convert to grayscale if needed
    if size(I,3) == 3
        I = rgb2gray(I);
    end

    % --- Convert to binary mask ---
    % White (255) → 255
    % Anything else → 0

    mask = I == 255;        % logical mask of pure white pixels
    out  = uint8(mask) * 255;

    % --- Save cleaned version ---
    imwrite(out, 'ref8_clean.png');

    % --- Optional debug display ---
    figure; 
    subplot(1,2,1); imshow(I, []); title('Original ref8.png');
    subplot(1,2,2); imshow(out,[]); title('Cleaned ref8 mask');

    fprintf('Saved cleaned mask as ref8_clean.png\n');
end
