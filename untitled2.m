clc;
clear;
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
%     img_origin_0=rgb2gray(img_origin_0);
    img_origin_0_R = img_origin_0(:,:,1);
    img_origin_0_G = img_origin_0(:,:,2);
    img_origin_0_B = img_origin_0(:,:,3);
    
    img_origin_45 =  imread(strcat(imgPath_45,image_name_45));
%     img_origin_45=rgb2gray(img_origin_45);
    img_origin_45_R = img_origin_45(:,:,1);
    img_origin_45_G = img_origin_45(:,:,2);
    img_origin_45_B = img_origin_45(:,:,3);
    
    img_origin_90 =  imread(strcat(imgPath_90,image_name_90));
%     img_origin_90=rgb2gray(img_origin_90);
    img_origin_90_R = img_origin_90(:,:,1);
    img_origin_90_G = img_origin_90(:,:,2);
    img_origin_90_B = img_origin_90(:,:,3);
    
    img_origin_135 =  imread(strcat(imgPath_135,image_name_135));
%     img_origin_135=rgb2gray(img_origin_135);
    img_origin_135_R = img_origin_135(:,:,1);
    img_origin_135_G = img_origin_135(:,:,2);
    img_origin_135_B = img_origin_135(:,:,3);
    I_R = 0.5 * ( double(img_origin_0_R) + double(img_origin_90_R) + double(img_origin_45_R) + double(img_origin_135_R));
    I_G = 0.5 * ( double(img_origin_0_G) + double(img_origin_90_G) + double(img_origin_45_G) + double(img_origin_135_G));
    I_B = 0.5 * ( double(img_origin_0_B) + double(img_origin_90_B) + double(img_origin_45_B) + double(img_origin_135_B));
    I_R = mat2gray(I_R);
    I_G = mat2gray(I_G);
    I_B = mat2gray(I_B);
    I(:,:,1) = I_R(:,:,1);
    I(:,:,2) = I_G(:,:,1);
    I(:,:,3) = I_B(:,:,1);
%     I_1_R = double(img_origin_0_R) + double(img_origin_90_R);
%     I_1_G = double(img_origin_0_G) + double(img_origin_90_G); 
%     I_1_B = double(img_origin_0_B) + double(img_origin_90_B);
%     I_1_R = mat2gray(I_1_R);
%     I_1_G = mat2gray(I_1_G);
%     I_1_B = mat2gray(I_1_B);
%     I_1(:,:,1) = I_1_R(:,:,1);
%     I_1(:,:,2) = I_1_G(:,:,1);
%     I_1(:,:,3) = I_1_B(:,:,1);
%     
%         
% %     I_2_R = double(img_origin_45_R) + double(img_origin_135_R);
% %     I_2_G = double(img_origin_45_G) + double(img_origin_135_G); 
% %     I_2_B = double(img_origin_45_B) + double(img_origin_135_B);
% %     
% %     I_2(:,:,1) = I_2_R(:,:,1);
% %     I_2(:,:,2) = I_2_G(:,:,1);
% %     I_2(:,:,3) = I_2_B(:,:,1);
    savedir_I = './S0/';
%     savedir_I1 = './S0_1/';
%     savedir_I2 = './train_S0_2_rgb/';
    savedname_I=fullfile(savedir_I,image_name_0);
    imwrite(I,savedname_I);colormap('jet');
%     savedname_I1=fullfile(savedir_I1,image_name_0);
%     savedname_I2=fullfile(savedir_I2,image_name_0);
%     imwrite(I_1,savedname_I1);colormap('jet');
%     imwrite(I_2,savedname_I2);colormap('jet');
%     i135=rgb2gray(img_origin_135);
%     i=0.5*(double(i0)+double(i90)+double(i45)+double(i135));
%     I=mat2gray(i);
%     imshow(I);
%     saveddir_I='./unpolar_rgb/';
%     savedname_I1=fullfile(saveddir_I,image_name_0);
%     imwrite(I,savedname_I);colormap('jet');
%     
    
%     imwrite(I,'I.PNG');    
%     q=double(i0)-double(i90);
%     u=double(i45)-double(i135);
% 
%     DOP=sqrt(q.*q+u.*u)./i;
%     DOP=sqrt(q.*q+u.*u);
%     DOP=mat2gray(DOP);
%     colormap('jet');
%     figure(1);imshow(DOP);colormap('jet');
%     saveddir_DOP='./test-new-dop/';
%     savedname_DOP=fullfile(saveddir_DOP,image_name_0);
%     imwrite(DOP,savedname_DOP);colormap('jet');
% 
%     aop=(1/2)*atan(u./q);
%     AOP=mat2gray(aop);
%     colormap('jet');
%     figure(2);imshow(AOP);colormap('jet');
%     saveddir_AOP='./all-aop/';
%     savedname_AOP=fullfile(saveddir_AOP,image_name_0);
%     imwrite(AOP,savedname_AOP);colormap('jet');
% 
%     figure(1);imshow(DOP); colormap('jet');
%     shading interp;
%     saveas(1,'DOP.jpg');
%     figure(2);imshow(AOP); colormap('jet');
%     shading interp;
%     saveas(2,'AOP.jpg');
%     saveas(AOP,fileSavePath_AOP);
%     saveas(DOP,fileSavePath_DOP);


end