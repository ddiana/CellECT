function run_seeded_waterhsed(input_mat_file, output_mat_file)

load (input_mat_file, 'vol', 'seeds','has_bg');

start_pts_mask = zeros(size(vol));


for i = 1:size(seeds,2)
	for x = -0:1
		for y = -0:1
			for z = -0:1
				xloc = max( round(seeds(1,i)+1) + x, 1);
				xloc = min( xloc, size(vol,1));

				yloc = max( round(seeds(2,i)+1) + y, 1);
				yloc = min( yloc, size(vol,2));

				zloc = max( round(seeds(3,i)+1) + z, 1);
				zloc = min( zloc, size(vol,3));
				
			    start_pts_mask(xloc, yloc,zloc) = 1;
			end
		end
	end
end

% if it has background, assume that this background surrounds the object of interest.
% Note: change this is not accurate
if has_bg
	start_pts_mask(:,1,:) = 1;
	start_pts_mask(:,end-1,:) = 1;
	start_pts_mask(1,:,:) = 1;
	start_pts_mask(end-1,:,:) = 1;
end

% if there are no nuclei (just dummy) and no background, then just return one big box

if (size(seeds,2) == 1) & (~has_bg)

	ws = ones(size(vol));

	ws(:,:,1) = zeros(size(vol,1), size(vol,2));
	ws(:,:,end) = zeros(size(vol,1), size(vol,2));
	ws(:,1,:) = zeros(size(vol,1),size(vol,3));
	ws(:,end,:) = zeros(size(vol,1),size(vol,3));
	ws(1,:,:) =  zeros(size(vol,2),size(vol,3));
	ws(end,:,:) =  zeros(size(vol,2),size(vol,3));

else
	% actually run watershed
	vol = imimposemin (vol, start_pts_mask);

	ws = watershed(vol);
end


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

