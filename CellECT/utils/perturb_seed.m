function [x,y,z] = perturb_seed (x,y,z, mask_to_avoid, max_img_size)

% 
% % perturn point if  in illegal position
% for idx = 1:size(x,2)
%     
%     if mask_to_avoid(x(idx), y(idx), z(idx)) ~=0
%         
%         not_found = true;
% 
%         box_x_min = x(idx);
%         box_x_max = x(idx);
%         box_y_min = y(idx);
%         box_y_max = y(idx);
%         box_z_min = z(idx);
%         box_z_max = z(idx);
% 
%         while not_found
%             box_x_min = max(box_x_min-1, 2);
%             box_x_max = min(box_x_max+1, max_img_size(1)-1);
% 
%             box_y_min = max(box_y_min-1, 2);
%             box_y_max = min(box_y_max+1, max_img_size(2)-1);
% 
%             box_z_min = max(box_z_min-1, 2);
%             box_z_max = min(box_z_max+1, max_img_size(3)-1);
% 
%             [perturbed_x, perturbed_y, perturbed_z] = find (mask_to_avoid(box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max) == 0);
% 
%             [x;y;z]
%             %size(mask_to_avoid(box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max))
%             if size(perturbed_x) >0
%                 x(idx) =  box_x_min + perturbed_x(1) -1;
%                 y(idx) =  box_y_min + perturbed_y(1) -1;
%                 z(idx) =  box_z_min + perturbed_z(1) -1;
%                 not_found = false;
%                 'perturbed'
%                 [x;y;z]
%             end
%         end
%     end
% end