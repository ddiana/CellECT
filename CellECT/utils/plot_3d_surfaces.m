labels = unique(ws);

padding = 10;
ws1 = zeros(size(ws) + padding*2);
ws1(padding+1:end-padding, padding+1:end-padding, padding+1:end-padding) = ws;

counter = 0;

figure;
lighting phong; 
camlight('headlight')

hold on
grid on

for lab = labels'
   counter = counter +1
   if lab >1
      mask = (ws1 == lab);
      mask = int8(convn (mask, ones(11,11,11), 'same'));
      h = patch(isosurface(mask, 50)); 
      set(h, 'FaceColor', [rand, rand, rand],'EdgeColor','none'); 

       
   end
end