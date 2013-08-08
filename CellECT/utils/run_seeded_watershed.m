function run_seeded_waterhsed(input_mat_file, output_mat_file)

load (input_mat_file, 'vol', 'seeds','has_bg');

start_pts_mask = zeros(size(vol));

for i = 1:size(seeds,2)
    start_pts_mask(round(seeds(1,i)+1), round(seeds(2,i)+1), round(seeds(3,i)+1)) = 1;
end

% if it has background, assume that this background surrounds the object of interest.
% Note: change this is not accurate
if has_bg
	start_pts_mask(:,1,:) = 1;
	start_pts_mask(:,end-1,:) = 1;
	start_pts_mask(1,:,:) = 1;
	start_pts_mask(end-1,:,:) = 1;
end

vol = imimposemin (vol, start_pts_mask);

ws = watershed(vol);

if ~has_bg
% if it doesnt have a bg, skip label 1 since this is reserved for background
	try
		ws = ws + uint16(ws>0);
	catch
		try
			ws = ws + uint8(ws>0);
		catch
			ws = ws + double(ws>0);
		end
	end
	
end

save (output_mat_file, 'ws');

quit

