rootdir='/scr/r6/tgebru/inverting_conv/caffe_invert_alexnet/data/pascal/VOCdevkit/VOC2012';
file_list=dir(fullfile(rootdir,'JPEGImages'));
in_list=cell(length(file_list),1);
for i=1:length(file_list)
  if length(file_list(i).name)<3
    continue
  end
  if (file_list(i).name(end-2:end)~='jpg')
    continue
  end
  
  in_list{i}=fullfile(rootdir,'JPEGImages',file_list(i).name);
end

in_list(i+1:end)=[];
empty=cellfun(@(x)isempty(x),in_list,'UniformOutput',false)
in_list(cell2mat(empty(:)))=[];

out_dir=fullfile(rootdir,'transformedImages/multFilters');

generate_image(in_list,false,out_dir);
