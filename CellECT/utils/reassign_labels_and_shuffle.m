function newvol = reassign_labels_and_shuffle(vol)

old_labels = unique(vol);

% remove background and border
old_labels(find(old_labels == 0)) = [];
old_labels(find(old_labels == 1)) = [];
old_labels = [0,1, old_labels'];

new_labels = randperm(numel(old_labels));
new_labels = new_labels+1;
new_labels = [0,1 new_labels];

label_assignment = ones(max(old_labels)+1,1) * -1;

counter = 1;
for i = 1:numel(old_labels)
    label_assignment(old_labels(i)+1) = new_labels(counter);
    counter = counter +1;
end

newvol = zeros(size(vol));

for k = 1:size(vol,3)
    k
    for j = 1:size(vol,2)
        for i = 1:size(vol,1)
            newvol(i,j,k) = label_assignment(vol(i,j,k)+1);
        end
    end
end