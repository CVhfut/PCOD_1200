clear;close all;clc;

imgPath_0 = ' ';% ͼ���·��
imgPath_45 = ' ';
imgPath_90 = ' ';
imgPath_135 = ' ';
% fileSavePath_AOP=' ';
% fileSavePath_DOP=' ';


img_path_list_0 = dir(strcat(imgPath_0,'*.jpg'));%��ȡ���ļ��������cjpg��ʽ��ͼ�w
img_path_list_45 = dir(strcat(imgPath_45,'*.jpg'));
img_path_list_90 = dir(strcat(imgPath_90,'*.jpg'));
img_path_list_135 = dir(strcat(imgPath_135,'*.jpg'));
img_num_0 = length(img_path_list_0);%��ȡͼ������
img_num_45 = length(img_path_list_45);
img_num_90 = length(img_path_list_90);
img_num_135 = length(img_path_list_135);

for T = 1:img_num_0
    pn_0 = 1:img_num_0;%��һ��ȡͼ��
    pn_45 = 1:img_num_45;
    pn_90 = 1:img_num_90;
    pn_135 = 1:img_num_135;

    image_name_0 = img_path_list_0(pn_0(T)).name;  
    image_name_45 = img_path_list_45(pn_45(T)).name;
    image_name_90 = img_path_list_90(pn_90(T)).name;
    image_name_135 = img_path_list_135(pn_135(T)).name;
    
    
    img_origin_0 =  imread(strcat(imgPath_0,image_name_0));%��ȡͼ��
    img_0_r=img_origin_0(:,:,1);
    img_0_g=img_origin_0(:,:,2);
    img_0_b=img_origin_0(:,:,3);
    img_0_r = repmat(img_0_r, [1,1,3]);
    img_0_g = repmat(img_0_g, [1,1,3]);
    img_0_b = repmat(img_0_b, [1,1,3]);
    i0_r=rgb2gray(img_0_r);
    i0_g=rgb2gray(img_0_g);
    i0_b=rgb2gray(img_0_b);
    i0=rgb2gray(img_origin_0);
    
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
    i45=rgb2gray(img_origin_45);
    
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
    i90=rgb2gray(img_origin_90);
    
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
    i135=rgb2gray(img_origin_135);
    
    i=0.5*(double(i0)+double(i90)+double(i45)+double(i135));  % S0
%     I=mat2gray(i);  % ������ת��Ϊ�Ҷ�ͼ��
    
    i_r=(double(i0_r)+double(i90_r)+double(i45_r)+double(i135_r));
    i_g=(double(i0_g)+double(i90_g)+double(i45_g)+double(i135_g));
    i_b=(double(i0_b)+double(i90_b)+double(i45_b)+double(i135_b));
    
    minR = min(i_r(:)); % ��ȡ���� i_r ����Сֵ
    maxR = max(i_r(:)); % ��ȡ���� i_r �����ֵ
    minG = min(i_g(:)); 
    maxG = max(i_g(:)); 
    minB = min(i_b(:)); 
    maxB = max(i_b(:)); 

    a_min = [minR,minG,minB];
    a_max = [maxR,maxG,maxB];
    Min = min(a_min);
    Max = max(a_max);

%     I_r = (i_r - Min)./(Max - Min);
%     I_g = (i_g - Min)./(Max - Min);
%     I_b = (i_b - Min)./(Max - Min);

%     I = cat(3,I_r,I_g,I_b);
    I = cat(3, i_r, i_g, i_b);

%     imshow(I);
%     saveddir_I='./S0_new/';
%     savedname_I=fullfile(saveddir_I,image_name_0);
%     imwrite(I,savedname_I);
%     colormap('jet');
    
    
    %imwrite(I,'I.PNG');    
    q=double(i0)-double(i90);
    u=double(i45)-double(i135);
    
    s1=mat2gray(q);
    s2=mat2gray(u);
    saveddir_s1='./S1_new/';
    savedname_s1=fullfile(saveddir_s1,image_name_0);
    imwrite(s1,savedname_s1);
    colormap('jet');
    saveddir_s2='./S2_new/';
    savedname_s2=fullfile(saveddir_s2,image_name_0);
    imwrite(s2,savedname_s2);
    colormap('jet');
 
    DOP=sqrt(q.*q+u.*u)./i;
  % DOP=sqrt(q.*q+u.*u);
    DOP=mat2gray(DOP);
    colormap('jet');
    figure(1);
%     imshow(DOP);
    colormap('jet');
    saveddir_DOP='./DOP_new/';
    savedname_DOP=fullfile(saveddir_DOP,image_name_0);
    imwrite(DOP,savedname_DOP);colormap('jet');

    aop=(1/2)*atan(u./q);
    AOP=mat2gray(aop);
    colormap('jet');
%     figure(2);imshow(AOP);colormap('jet');
    saveddir_AOP='./AOP_new/';
    savedname_AOP=fullfile(saveddir_AOP,image_name_0);
    imwrite(AOP,savedname_AOP);colormap('jet');

%     figure(1);imshow(DOP); colormap('jet');
%     shading interp;
%     saveas(1,'DOP.jpg');
%     figure(2);imshow(AOP); colormap('jet');
%     shading interp;
%     saveas(2,'AOP.jpg');
%     saveas(AOP,fileSavePath_AOP);
%     saveas(DOP,fileSavePath_DOP);
    del_i = I .* DOP;
    IP=mat2gray(del_i);
    colormap('jet');
%     figure(2);imshow(AOP);colormap('jet');
%     saveddir_IP=' ';
%     savedname_IP=fullfile(saveddir_IP,image_name_0);
%     imwrite(IP,savedname_IP);colormap('jet');


end