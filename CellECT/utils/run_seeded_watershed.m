function run_seeded_waterhsed(input_mat_file, output_mat_file)


debug = false;
display3d = false;

h_sig = [7,7,3];
h_size = [19,19,19];
[h_x,h_y,h_z] = ndgrid(-h_size(1):h_size(1), -h_size(2):h_size(2),-h_size(3):h_size(3));
h = exp(-(h_x.*h_x/2/h_sig(1)^2 + h_y.*h_y/2/h_sig(2)^2 + h_z.*h_z/2/h_sig(3)^2));
h = h/sum(h(:));
h = max(h(:)) - h;
h = h / max(h(:)) + 0.01;
h = h / max(h(:));


[mx, my, mz] = ndgrid(1 : 5, 1:5, 1:5);
sphere_mask = ((mx - 3 ) .^2 + (my-3).^2 + (mz- 3).^2  <5);

% 
% p = path;
% path(p, [pwd, '/fast_marching']);

bg_mask = [];

load (input_mat_file, 'vol', 'sbx', 'sby', 'sbz','seeds', 'bg_mask');
seeds = seeds(:);

background_seeds = [sbx(:), sby(:), sbz(:)];

number_seeds = size(seeds,1) -2;

start_pts_mask = zeros(size(vol));

if size(bg_mask,1)==0
    bg_mask = zeros(size(start_pts_mask));
end

bg_mask = cast(bg_mask, class(vol));

radius = 40;

[mx, my, mz] = ndgrid(1 : radius*2 +2 , 1:radius*2 +2, 1:radius*2 +2);

mask_neighborhood = ((mx - radius -1) .^2 + (my-radius -1).^2 + (mz-radius -1).^2 > radius^2);

% if ~strcmp(class(seeds),'cell')
% 	seeds = squeeze(seeds);
% % 	if size(seeds,1) == 3
% % 		seeds = seeds';
% % 	end
% end



for i = 1:number_seeds
    seed_group = floor(seeds{i} + 1);


    
    if size(seed_group,1) >1
        min_box = min(seed_group,[],1);
        max_box = max(seed_group,[],1);
        min_box = max(min_box - 10, 1);
        max_box = min(max_box + 10, cast(size(vol), class(max_box)));
        
        one =  seed_group(:,1) - min_box(1)+1;
        two =  seed_group(:,2) - min_box(2)+1;
        three = seed_group(:,3) - min_box(3) +1;
 
        
        for idx = 1:size(one,1)
            [one, two, three] = perturb_seed(one, two, three, start_pts_mask);
            bg_mask = paste_mask_in_vol(bg_mask, mask_neighborhood, [one(idx), two(idx), three(idx)]);
        end
        
        input = [ one, two, three ];
        input = double(input);        
        
        output = connect_seeds(vol(min_box(1):max_box(1), min_box(2):max_box(2), min_box(3):max_box(3)), input, start_pts_mask(min_box(1):max_box(1), min_box(2):max_box(2), min_box(3):max_box(3)), min_box, bg_mask(min_box(1):max_box(1), min_box(2):max_box(2), min_box(3):max_box(3)));
        start_pts_mask(min_box(1):max_box(1), min_box(2):max_box(2), min_box(3):max_box(3)) = output;
    else
        
        [new_x, new_y, new_z] = perturb_seed([seed_group(1)], [seed_group(2)], [seed_group(3)], start_pts_mask);
        bg_mask = paste_mask_in_vol(bg_mask, mask_neighborhood, [new_x(1), new_y(1), new_z(1)]);
    
        %vol = paste_mask_in_vol(vol, h, [new_x(1), new_y(1), new_z(1)]);
        
        start_pts_mask(new_x(1), new_y(1), new_z(1)) = 1;     
        if debug
            plot3(new_x(1), new_y(1), new_z(1) ,'k.','markersize',15);
            hold on
        end
    end
          
        
        
% 	for x = -0:1
% 		for y = -0:1
% 			for z = -0:1
% 				xloc = max( round(seeds(1,i)+1) + x, 1);
% 				xloc = min( xloc, size(vol,1));
% 
% 				yloc = max( round(seeds(2,i)+1) + y, 1);
% 				yloc = min( yloc, size(vol,2));
% 
% 				zloc = max( round(seeds(3,i)+1) + z, 1);
% 				zloc = min( zloc, size(vol,3));
% 				
% 			    start_pts_mask(xloc, yloc,zloc) = 1;
% 			end
% 		end
%     end
end


for i = 1:size(background_seeds,1)
	background_seeds(i,1) = max( round(background_seeds(i,1)+1) , 1);
    xloc = background_seeds(i,1) ;
	xloc = min( xloc, size(vol,1));

	background_seeds(i,2) = max( round(background_seeds(i,2)+1) , 1);
    yloc = background_seeds(i,2);
	yloc = min( yloc, size(vol,2));

    background_seeds(i,3) = max( round(background_seeds(i,3)+1) , 1);
    zloc = background_seeds(i,3);
	zloc = min( zloc, size(vol,3));
    
    [xloc, yloc, zloc] = perturb_seed([xloc], [yloc], [zloc], start_pts_mask);
				
    xloc = xloc(1);
    yloc = yloc(1);
    zloc = zloc(1);
    
    start_pts_mask(xloc, yloc,zloc) = 1;
    
    % scale vol around bg_seeds:
    %vol = paste_mask_in_vol(vol, h, [xloc, yloc, zloc]);
        
    
end

bg_mask_sum = sum(bg_mask(:));

has_bg = (size(background_seeds,1)>0) | (bg_mask_sum>0);


% if it has background, assume that this background surrounds the object of interest.
% Note: change this is not accurate
% if has_bg
% 	start_pts_mask(:,1,:) = 1;
% 	start_pts_mask(:,end-1,:) = 1;
% 	start_pts_mask(1,:,:) = 1;
% 	start_pts_mask(end-1,:,:) = 1;
% end

% if there are no nuclei (just dummy) and no background, then just return one big box

% if (size(seeds,2) == 1) & (~has_bg)
% 
% 	ws = ones(size(vol));
% 
% 	ws(:,:,1) = zeros(size(vol,1), size(vol,2));
% 	ws(:,:,end) = zeros(size(vol,1), size(vol,2));
% 	ws(:,1,:) = zeros(size(vol,1),size(vol,3));
% 	ws(:,end,:) = zeros(size(vol,1),size(vol,3));
% 	ws(1,:,:) =  zeros(size(vol,2),size(vol,3));
% 	ws(end,:,:) =  zeros(size(vol,2),size(vol,3));
% 
% else
% 	% actually run watershed
% 	vol = imimposemin (vol, start_pts_mask);
% 
% 	ws = watershed(vol);
% end

bg_mask = cast(bg_mask, class(start_pts_mask));


if (number_seeds < 1)
    ws = ones(size(vol));
else
    if ((number_seeds >= 1) & ( has_bg ) ) | (number_seeds>1)
        % run watershed if at least one other background seed
        vol = imimposemin (vol, start_pts_mask + bg_mask);
        ws = watershed(vol);
    else
        % make everythign label 1 (this will get boosted to 2 later)
        ws =  ones(size(vol));
    end
end


% move segment labels to starts at 2, because 1 will be given to bg
% segments

mask = cast(ws >=1, class(ws));
ws = ws+mask; 


% make label 1 for everything background from connected components in
% bg_mask

if bg_mask_sum
   
    labels = unique(cast(bg_mask, class(ws)) .* ws);
    for label = labels'
       if label ~=0
           mask = cast(ws == label, class(ws));
           ws = mask + (1-mask).*ws;
       end
    end
    
end


% make label 1 for everything that is background from seeds
for i = 1:size(background_seeds,1)
    label = ws(background_seeds(i,1), background_seeds(i,2), background_seeds(i,3));
    if label ~=0
        mask = cast(ws == label, class(ws));
        ws = mask + (1-mask).*ws;
    end
end




% if (~has_bg) && (size(seeds,2)>0)
% % if it doesnt have a bg, skip label 1 since this is reserved for background
% 	try
% 		ws = ws + uint16(ws>0);
% 	catch
% 		try
% 			ws = ws + uint8(ws>0);
% 		catch
% 			ws = ws + double(ws>0);
% 		end
% 	end
% end

% remove background boundary
if (size(background_seeds,1)>1) | (bg_mask_sum)
    %mask = (ws > 1);
    
    near_non1s = convn(logical (ws > 1), sphere_mask, 'same');
    mask = (near_non1s >0);
    %mask = convn(logical(mask),[1 1 1;1 1 1;1 1 1],'same')>=1;
    %mask = convn(logical(mask),[1 1 1;1 1 1;1 1 1],'same')>=1;
    mask = cast(mask, class(ws));
    
    ws = ws .* mask + (1-mask);
    
end

save (output_mat_file, 'ws');

if display3d
    figure
    [mx,my,mz] = meshgrid(0:1:size(ws,1)-1, 0:1:size(ws,2)-1, 0:1:size(ws,3)-1);
    recolored = reassign_labels_and_shuffle(ws);
    recolored = double(recolored<2)*double(max(ws(:)+1)) + recolored.* double(recolored>=2);
    xslice = [160];
    yslice = [];
    zslice = [70];
    slice(mx,my,mz,recolored, xslice, yslice, zslice);
    colormap colorcube
    shading interp
    colormapeditor
    
    figure
    slice(mx,my,mz,vol, xslice, yslice, zslice);
    colormap hot
    shading interp
end

quit

