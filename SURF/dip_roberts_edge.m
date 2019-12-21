function [edge_img] = dip_roberts_edge(img,thresh)
% This is our awesome dip (wtf is dip?) function
% Inputs:
% 1. Original image
% 2. threshold: the threshold is the lower bound of the gradient which we
% accept as edge. any gradient value below the threshold becomes zero,
% meaning we do not consider it as an edge.
%Output: Edge image, which is the gradient of the original image, after
%applying thresholding. the outcome is a BW image, in order to better
%understand the edges we found
% 

gx = 0.5 * [1 0 ; 0 -1];
gy = 0.5 * [0 1 ; -1 0];

edgeGx = conv2(img , gx , 'same');
edgeGy = conv2(img , gy , 'same');

gradEdge = sqrt(edgeGx.^2 + edgeGy.^2);
% angleEdge = atan(edgeGy./edgeGx);

gradEdge(gradEdge < thresh) = 0;

gradEdge(gradEdge ~= 0) = 1;

edge_img = gradEdge;
end

