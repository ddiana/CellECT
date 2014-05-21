function  output = connect_seeds(I, pts, mask_to_avoid_copy, min_box, bg_mask )

debug = false;

mask_to_avoid = mask_to_avoid_copy;
output = mask_to_avoid_copy;
%mask_to_avoid = (mask_to_avoid + bg_mask)>0;
min_box = double(min_box);

x = pts(:,1)';
y = pts(:,2)';
z = pts(:,3)';

xmean = mean(x);
ymean = mean(y);
zmean = mean(z);

dist = (x-xmean).^2 + (y-ymean).^2 + (z-zmean).^2;
[val, pos] = min(dist);

I = reshape(awgn(double(I(:)), 0.1), size(I)) ;
I = abs(I);

flt = fspecial ( 'gaussian', [5, 5], 3);  

I1 = imfilter(I+0.0001, flt);


I1 = max(I1(:)) - I1 ;

if max(mask_to_avoid(:))>0
    mask_to_avoid = convn(logical(mask_to_avoid),[1 1 1;1 1 1;1 1 1],'same')>=1;
    mask_to_avoid = cast(mask_to_avoid, class(I1));

    I1 = I1.* (1 - mask_to_avoid);
end

%imagesc(I1);


% TODO: round out of range

 
    

options.nb_iter_max = Inf;
[D,S] = perform_fast_marching(I1, [x(pos); y(pos); z(pos)], options);

% forcing margins to be inf so that the path doesnt exit the matrix:
D(:,:,end) = Inf;
D(:,:,1) = Inf;
D(:,end,:) = Inf;
D(:,1,:) = Inf;
D(end,:,:) = Inf;
D(1,:,:) = Inf;




cc = hsv(100);

r = randi(100);

for i = [1:pos-1, pos+1:size(x,2)]
    path = compute_geodesic(D, [x(i); y(i); z(i)]);
    if debug
        plot3(path(1,:)+ min_box(1), path(2,:)+ min_box(2), path(3,:)+ min_box(3), 'linewidth', 3, 'color', cc(r,:) );
        hold on
    end
    path = round(path);
    
    for idx = 1:size(path,2)
        xval = path(1,idx);
        yval = path(2,idx);
        zval = path(3,idx);
        output(xval, yval, zval) = 1;    
    end
    


    %plot( path(2,:), path(1,:), 'k' , 'linewidth', 3);
end

if debug
plot3(x + min_box(1), y+ min_box(2), z+ min_box(3) ,'k.','markersize',15);
grid
end

% 
% for i = 6:15
%     I(x(i), y(i)) =0;
% end
% 
% Iws = imimposemin(I, I==0);
% 
% result = watershed(Iws);
% 
% figure
% imagesc(result);