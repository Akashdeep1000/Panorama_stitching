function x = image_panorama_custom()
location = './data_custom';
ds = imageDatastore(location);
indices = 1:5;

subimds = subset(ds, indices);

im1 = readimage(subimds,1);
img1 = imresize(im1,[600 800]);
im2 = readimage(subimds,2);
img2 = imresize(im2,[600 800]);
im3 = readimage(subimds,3);
img3 = imresize(im3,[600 800]);
im4 = readimage(subimds,4);
img4 = imresize(im4,[600 800]);
im5 = readimage(subimds,5);
img5 = imresize(im5,[600 800]);

Originalimg = {img1, img2, img3, img4, img5};
figure(1) ; 
clf ;
montage({img1,img2,img3,img4,img5});

numImg=5;

% convert rgb into grayscale and single
image1 = rgb2gray(im2single(img1)) ;
image2 = rgb2gray(im2single(img2)) ;
image3 = rgb2gray(im2single(img3)) ;
image4 = rgb2gray(im2single(img4)) ;
image5 = rgb2gray(im2single(img5)) ;
GrayImg = {image1, image2, image3, image4, image5} ;

%keypoints for each image and save it.
for j=1:numImg 
   [fea,des]= vl_sift(GrayImg{j}) ;
   features{j}=fea; descriptors{j}=des;
end


%start with middle Image
Imagemid=ceil(numImg/2); 
rotInd=floor(numImg/2);

%match first 2 images. 

if mod(numImg,2)==0
    imgIndex1= Imagemid-1; imgIndex2=Imagemid;
    [X1,X2, matches] = matchkeyimg(features{imgIndex1},descriptors{imgIndex1},features{imgIndex2},descriptors{imgIndex2});
    H=ransac(X1,X2,matches);
    pano = panostitch(H,Originalimg{imgIndex1},Originalimg{imgIndex2});
    mosaicSingle = rgb2gray(im2single(pano)) ;
    indicesToIgnore = -1; 
else
    imgIndex1= Imagemid; imgIndex2=Imagemid+1;
    [X1,X2, matches] = matchkeyimg(features{imgIndex1},descriptors{imgIndex1},features{imgIndex2},descriptors{imgIndex2});
    H=ransac(X1,X2,matches);
    pano = panostitch(H,Originalimg{imgIndex1},Originalimg{imgIndex2});
    mosaicSingle = rgb2gray(im2single(pano)) ;   
    indicesToIgnore = 1;
end

%match the remianing images
for j=1:rotInd
   for k = [-1,1] %stitching right and left.
       imgInd=Imagemid + (j*k); 
       if (j*k)~=indicesToIgnore && (imgInd~=0) 
           
            [fea,des] = vl_sift(mosaicSingle);
            [X1,X2, matches] = matchkeyimg(fea,des,features{imgInd},descriptors{imgInd});
            H=ransac(X1,X2,matches);
            pano = panostitch(H,pano,Originalimg{imgInd});
            mosaicSingle = rgb2gray(im2single(pano)) ;
       end
   end
end

figure(2) ; clf ;
imagesc(pano) ; axis image off ;
title('Panorama') ;
imwrite(pano,'../Panorama_custom.png');

%keypoints matching.
    function [X1,X2,matches] = matchkeyimg(f1,d1,f2,d2)
    [matches, ~] = vl_ubcmatch(d1,d2) ;
    X1 = f1(1:2,matches(1,:)) ; X1(3,:) = 1 ; 
    X2 = f2(1:2,matches(2,:)) ; X2(3,:) = 1 ; 
end

% panaroma stitching.
    function pano = panostitch(H,im1,im2)
    box2 = [1  size(im2,2) size(im2,2)  1 ;
        1  1           size(im2,1)  size(im2,1) ;
        1  1           1            1 ] ;
    box2_ = H \ box2 ;
    box2_(1,:) = box2_(1,:) ./ box2_(3,:) ;
    box2_(2,:) = box2_(2,:) ./ box2_(3,:) ;
    ur = min([1 box2_(1,:)]):max([size(im1,2) box2_(1,:)]) ;
    vr = min([1 box2_(2,:)]):max([size(im1,1) box2_(2,:)]) ;

    [u,v] = meshgrid(ur,vr) ;
    im1_ = vl_imwbackward(im2double(im1),u,v) ;

    z_ = H(3,1) * u + H(3,2) * v + H(3,3) ;
    u_ = (H(1,1) * u + H(1,2) * v + H(1,3)) ./ z_ ;
    v_ = (H(2,1) * u + H(2,2) * v + H(2,3)) ./ z_ ;
    im2_ = vl_imwbackward(im2double(im2),u_,v_) ;

    mass = ~isnan(im1_) + ~isnan(im2_) ;
    im1_(isnan(im1_)) = 0 ;
    im2_(isnan(im2_)) = 0 ;
    pano = (im1_ + im2_) ./ mass ;
end
%Ransac with homography.
function H = ransac(X1,X2,matches)
    clear H score ok ;
    nMatches = size(matches,2) ;
    for t = 1:100
        %get 4 points
        sub = vl_colsubset(1:nMatches, 4) ;
        A = [] ;
        %homography matrix.
        for i = sub
            A = cat(1, A, kron(X1(:,i)', vl_hat(X2(:,i)))) ;    
        end
        [~,~,V] = svd(A) ; 
        %last column after doing SVD.
        H{t} = reshape(V(:,9),3,3) ; 

      % homography
      X2_ = H{t} * X1 ; 
      du = X2_(1,:)./X2_(3,:) - X2(1,:)./X2(3,:) ;
      dv = X2_(2,:)./X2_(3,:) - X2(2,:)./X2(3,:) ;
      ok{t} = (du.*du + dv.*dv) < 5*5 ; % if distance < threshold.
      score(t) = sum(ok{t}) ;
    end
    %step 4 - find the best one.
    [~, best] = max(score) ;
    H = H{best} ;
end
end