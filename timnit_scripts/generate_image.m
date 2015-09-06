function generate_image(filelist, apply_blur, out_dir)
% This function applies multiple color filters to each image.
%
% filelist = {'file1', 'file2'};
% out_dir = '/dir/to/save/file';
% generate_image(filelist, true, out_dir);

    if (nargin < 1) || isempty(filelist)
        filelist{1} = 'http://tacolicious.com/wp-content/uploads/2013/02/Stanford.jpg';
        filelist{2} = 'http://jadeluckclub.com/wp-content/uploads/2011/11/MIT.jpg';
    end
    if nargin < 3
        out_dir = [];
    end

    rand('seed', 0);
    
    % Filter setup (if you want, you can put this inside the loop, which
    % will create random filters for each input image)
    FILTER_N = 10;
    filters = rand(FILTER_N,3); % first 3 columns deal with color channels
    if apply_blur
        filters(:,4) = rand(FILTER_N, 1)*3; % last column is a gaussian blur sigma
    else
        filters(:,4) = 0;
    end
    
    parfor i = 1:length(filelist)
        filelist{i}
        im_in = imread(filelist{i});
        im_in = double(im_in)/255;
        
        [~,filename,fileext] = fileparts(filelist{i});        
        
        im1 = rgb2hsv(colorfilter(im_in, [150 280]));
        im2 = rgb2hsv(colorfilter(im_in, [280 50]));
        im3 = rgb2hsv(colorfilter(im_in, [50 150]));
        
        for filter_i = 1:size(filters,1)
            suffix = sprintf('_%.3f', filters(filter_i,:));
            outname = sprintf('%s/%s%s%s', out_dir, filename, suffix, fileext);

            % color channel
            im_out = rgb2hsv(im_in);
            im_out(:,:,2) = im1(:,:,2)*filters(filter_i,1) + im2(:,:,2)*filters(filter_i,2) + im3(:,:,2)*filters(filter_i,3);
            im_out = hsv2rgb(im_out);
            
            % blurring
            if filters(filter_i,4) > 0
                H = fspecial('gaussian', 10, filters(filter_i,4));
                im_out = imfilter(im_out, H, 'replicate');
            end
        
            if isempty(out_dir)
                image(im_out);
                pause;
            else
                imwrite(im_out, outname);
            end
        end
        
        fprintf('%d/%d done.\n', i, length(filelist));
    end
end

function I = colorfilter(image, range)
    I = rgb2hsv(image);
    
    range = range./360;
    
    if(range(1) > range(2))
        mask = (I(:,:,1)>range(1) & (I(:,:,1)<=1)) + (I(:,:,1)<range(2) & (I(:,:,1)>=0));
    else
        mask = (I(:,:,1)>range(1)) & (I(:,:,1)<range(2));
    end
    
    I(:,:,2) = mask .* I(:,:,2);
    I = hsv2rgb(I);    
end
