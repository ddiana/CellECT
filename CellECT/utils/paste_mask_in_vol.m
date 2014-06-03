function vol = paste_mask_in_vol(vol, mask, center)

    radius = floor(size(mask,1)/2);
    rx_m = min(center(1) , radius) -1; 
    rx_p = min(size(vol,1) - center(1) , radius) ;
    
    ry_m = min(center(2) , radius) -1;
    ry_p = min(size(vol,2) - center(2) , radius) ;
    
    rz_m = min(center(3) , radius) -1;
    rz_p = min(size(vol,3) - center(3) , radius) ;
    
    mid = floor(size(mask,1)/2);
    vol(center(1) - rx_m: center(1) + rx_p, center(2) - ry_m : center(2) + ry_p, center(3) - rz_m : center(3) + rz_p) = vol(center(1) - rx_m: center(1) + rx_p, center(2) - ry_m : center(2) + ry_p, center(3) - rz_m : center(3) + rz_p) .* mask(mid-rx_m: mid+rx_p, mid-ry_m:mid+ry_p, mid-rz_m:mid+rz_p);
    
    