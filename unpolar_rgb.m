imgPath_0 = ' ';% 图像库路径
imgPath_45 = ' ';
imgPath_90 = ' ';
imgPath_135 = ' ';
% fileSavePath_AOP=' ';
% fileSavePath_DOP=' ';


img_path_list_0 = dir(strcat(imgPath_0,'*.jpg'));%获取该文件夹中所有.jpg格式的图像
img_path_list_45 = dir(strcat(imgPath_45,'*.jpg'));
img_path_list_90 = dir(strcat(imgPath_90,'*.jpg'));
img_path_list_135 = dir(strcat(imgPath_135,'*.jpg'));
img_num_0 = length(img_path_list_0);%获取图像总数
img_num_45 = length(img_path_list_45);
img_num_90 = length(img_path_list_90);
img_num_135 = length(img_path_list_135);

for T= 1:1200
    pn_0 = 1:img_num_0;%逐一读取图像
    pn_45 = 1:img_num_45;
    pn_90 = 1:img_num_90;
    pn_135 = 1:img_num_135;

    image_name_0 = img_path_list_0(pn_0(T)).name;% 图像名
    image_name_45 = img_path_list_45(pn_45(T)).name;
    image_name_90 = img_path_list_90(pn_90(T)).name;
    image_name_135 = img_path_list_135(pn_135(T)).name;
    
    
    img_origin_0 =  imread(strcat(imgPath_0,image_name_0));%读取图像
    img_0_r=img_origin_0(:,:,1);
    img_0_g=img_origin_0(:,:,2);
    img_0_b=img_origin_0(:,:,3);
    img_0_r = repmat(img_0_r, [1,1,3]);
    img_0_g = repmat(img_0_g, [1,1,3]);
    img_0_b = repmat(img_0_b, [1,1,3]);
    i0_r=rgb2gray(img_0_r);
    i0_g=rgb2gray(img_0_g);
    i0_b=rgb2gray(img_0_b);
    
    img_origin_45 =  imread(strcat(imgPath_45,image_name_45));
    img_45_r=img_origin_45(:,:,1);
    img_45_g=img_origin_45(:,:,2);
    img_45_b=img_origin_45(:,:,3);
    img_45_r = repmat(img_45_r, [1,1,3]);
    img_45_g = repmat(img_45_g, [1,1,3]);
    img_45_b = repmat(img_45_b, [1,1,3]);
    i45_r=rgb2gray(img_45_r);
    i45_g=rgb2gray(img_45_g);
    i45_b=rgb2gray(img_45_b);
    
    img_origin_90 =  imread(strcat(imgPath_90,image_name_90));
    img_90_r=img_origin_90(:,:,1);
    img_90_g=img_origin_90(:,:,2);
    img_90_b=img_origin_90(:,:,3);
    img_90_r = repmat(img_90_r, [1,1,3]);
    img_90_g = repmat(img_90_g, [1,1,3]);
    img_90_b = repmat(img_90_b, [1,1,3]);
    i90_r=rgb2gray(img_90_r);
    i90_g=rgb2gray(img_90_g);
    i90_b=rgb2gray(img_90_b);
    
    img_origin_135 =  imread(strcat(imgPath_135,image_name_135));
    img_135_r=img_origin_135(:,:,1);
    img_135_g=img_origin_135(:,:,2);
    img_135_b=img_origin_135(:,:,3);
    img_135_r = repmat(img_135_r, [1,1,3]);
    img_135_g = repmat(img_135_g, [1,1,3]);
    img_135_b = repmat(img_135_b, [1,1,3]);
    i135_r=rgb2gray(img_135_r);
    i135_g=rgb2gray(img_135_g);
    i135_b=rgb2gray(img_135_b);
    
%     i_r=0.5*(double(i0_r)+double(i90_r)+double(i45_r)+double(i135_r));
%     i_g=0.5*(double(i0_g)+double(i90_g)+double(i45_g)+double(i135_g));
%     i_b=0.5*(double(i0_b)+double(i90_b)+double(i45_b)+double(i135_b));
    
%     S1_r = abs(double(i0_r) - double(i90_r));
%     S1_g = abs(double(i0_g) - double(i90_g));
%     S1_b = abs(double(i0_b) - double(i90_b));
%     S1_r = mat2gray(S1_r);
%     S1_g = mat2gray(S1_g);
%     S1_b = mat2gray(S1_b);
%     
    S2_r = (double(i45_r) - double(i135_r));
    S2_g = (double(i45_g) - double(i135_g));
    S2_b = (double(i45_b) - double(i135_b));
    S2_r = mat2gray(S2_r);
    S2_g = mat2gray(S2_g);
    S2_b = mat2gray(S2_b);

%     
%     dop_r = sqrt(S1_r.*S1_r + S2_r.*S2_r)./i_r;
%     dop_g = sqrt(S1_g.*S1_g + S2_r.*S2_g)./i_g;
%     dop_b = sqrt(S1_b.*S1_b + S2_r.*S2_b)./i_b;
%     
%     aop_r = (1/2)*atan(S2_r./S1_r);
%     aop_g = (1/2)*atan(S2_g./S1_g);
%     aop_b = (1/2)*atan(S2_b./S1_b);
       
%     I_r=mat2gray(i_r);    
%     I_g=mat2gray(i_g);
%     I_b=mat2gray(i_b);
    
%     AOP_r = mat2gray(aop_r);
%     AOP_g = mat2gray(aop_g);
%     AOP_b = mat2gray(aop_b);
%     
%     DOP_r = mat2gray(dop_r);
%     DOP_g = mat2gray(dop_g);
%     DOP_b = mat2gray(dop_b);
%    
%     imshow(I);
%     I=cat(3,I_r,I_g,I_b);
%       S1 = cat(3,S1_r, S1_g, S1_b);
      S2 = cat(3,S2_r, S2_g, S2_b);
%     AOP=cat(3,AOP_r,AOP_g,AOP_b);
%     DOP=cat(3,DOP_r,DOP_g,DOP_b);
%     saveddir_I = ' ';
%     savedir_I2 = ' ';
%     saveddir_I=' ';
    saveddir_S2 = ' ';
    savedname_S2=fullfile(saveddir_S2,image_name_0);
    imwrite(S2,savedname_S2);colormap('jet');
    
end