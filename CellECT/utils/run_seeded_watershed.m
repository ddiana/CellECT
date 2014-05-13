function run_seeded_waterhsed(input_mat_file, output_mat_file)

debug = false;

p = path;
path(p, [pwd, '/fast_marching']);

load (input_mat_file, 'vol', 'seeds', 'bg_mask');

background_seeds = [];

try
    load(input_mat_file, 'background_seeds');
catch
end
        

start_pts_mask = [];
try
start_pts_mask = bg_mask
catch
start_pts_mask = zeros(size(vol));
end


if ~strcmp(class(seeds),'cell')
	seeds = squeeze(seeds);
	if size(seeds,1) == 3
		seeds = seeds';
	end
end



for i = 1:size(seeds,1)
   	seed_group = [];
    if strcmp(class(seeds),'cell')
        seed_group = floor(seeds{i} + 1);
    else
        seed_group = floor(seeds(i,:) +1);
    end
    if size(seed_group,1) >1
        min_box = min(seed_group,[],1);
        max_box = max(seed_group,[],1);
        min_box = max(min_box - 10, 1);
        max_box = min(max_box + 10, cast(size(vol), class(max_box)));
        one =  seed_group(:,1) - min_box(1)+1;
        two =  seed_group(:,2) - min_box(2)+1;
        three = seed_group(:,3) - min_box(3) +1;
        input = [ one, two, three ];
        input = double(input);
        output = connect_seeds(vol(min_box(1):max_box(1), min_box(2):max_box(2), min_box(3):max_box(3)), input, start_pts_mask(min_box(1):max_box(1), min_box(2):max_box(2), min_box(3):max_box(3)), min_box);
        start_pts_mask(min_box(1):max_box(1), min_box(2):max_box(2), min_box(3):max_box(3)) = output;
    else
        start_pts_mask(seed_group(1), seed_group(2), seed_group(3)) = 1;     
        if debug
            plot3(seed_group(1), seed_group(2), seed_group(3) ,'k.','markersize',15);
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
				
    start_pts_mask(xloc, yloc,zloc) = 1;
end



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

if (size(seeds,1) <= 1)
    ws = ones(size(vol));
else
    vol = imimposemin (vol, start_pts_mask);
    ws = watershed(vol);
end

mask = cast(ws >=1, class(ws));
ws = ws+mask; 

% make label 1 for everything that is background
for i = 1:size(background_seeds,1)
    label = ws(background_seeds(i,1), background_seeds(i,2), background_seeds(i,3));
    if label ~=0
        mask = cast(ws == label, class(ws));
        ws = mask + (1-mask).*ws;
    end
end

has_bg = (size(background_seeds,1)>0);

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
if size(background_seeds,1)>0
    mask = (ws > 1);
    
    mask = convn(logical(mask),[1 1 1;1 1 1;1 1 1],'same')>=1;
    mask = convn(logical(mask),[1 1 1;1 1 1;1 1 1],'same')>=1;
    mask = cast(mask, class(ws));
    
    ws = ws .* mask + (1-mask);
    
end

save (output_mat_file, 'ws');

quit

