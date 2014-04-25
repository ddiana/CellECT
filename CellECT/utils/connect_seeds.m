function  output = connect_seeds(I, pts)

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


%I1 = max(I1(:)) - I1 ;

%imagesc(I1);


% TODO: round out of range


options.nb_iter_max = Inf;
[D,S] = perform_fast_marching(I1, [x(pos); y(pos); z(pos)], options);

output = zeros(size(I));

for i = [1:pos-1, pos+1:size(x,2)]
    path = round(compute_geodesic(D, [x(i); y(i); z(i)]));
    for idx = 1:size(path,2)
        xval = path(1,idx);
        yval = path(2,idx);
        zval = path(3,idx);
        output(xval, yval, zval) = 1;    
    end
    
    %plot( path(2,:), path(1,:), 'k' , 'linewidth', 3);
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